from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from simnibs import mesh_io, transformations

# All paths and constants live in the repo-root .env (see .env.example).
ENV_FILE = Path(__file__).resolve().parents[1] / '.env'

def load_env(path):
    """Parse KEY=VALUE lines; '#' starts a comment line; surrounding quotes are stripped."""
    if not path.is_file():
        raise FileNotFoundError(f'Config file not found: {path}\nCopy .env.example to .env and edit it.')
    env = {}
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'): continue
        key, sep, value = line.partition('=')
        if not sep: raise ValueError(f'{path}:{n}: expected KEY=VALUE, got {line!r}')
        env[key.strip()] = value.strip().strip('"\'')
    return env

CFG = load_env(ENV_FILE)

def cfg(key):
    if key not in CFG: raise KeyError(f'{key} missing from {ENV_FILE}')
    return CFG[key]

def cfg_vector(key):
    return np.array([float(v) for v in cfg(key).split(',')])

FILE_SHARE = Path(cfg('FILE_SHARE'))
INPUT = FILE_SHARE / cfg('INPUT_DIR')
OUTPUT = FILE_SHARE / cfg('OUTPUT_DIR')
OUTPUT_STEM = cfg('OUTPUT_STEM')
HEAD_DIR = Path(cfg('HEAD_DIR'))
NAMES = ('P1','P2','P3')
MESH_PATHS = {k: INPUT / cfg(f'{k}_MSH') for k in NAMES}
TARGET_MNI = cfg_vector('TARGET_MNI')
ROI_RADIUS_MM = float(cfg('ROI_RADIUS_MM'))
GM_TAG = int(cfg('GM_TAG'))
SCALP_TAG = int(cfg('SCALP_TAG'))
SEED = int(cfg('SEED'))
# Relative scaling of Rafael's original dI/dt, not amperes.
# Nonnegative bounds model same-polarity synchronized excitation.
REFERENCE_DIDT = float(cfg('REFERENCE_DIDT_A_PER_S'))
COIL_MIN = float(cfg('COIL_MIN'))
COIL_MAX = float(cfg('COIL_MAX'))
MAX_SUM = float(cfg('MAX_SUM'))   # shared normalized drive budget; exploratory, not a hardware rating
BASELINE = cfg_vector('BASELINE_WEIGHTS')
SURFACE_LIMIT_RATIO = float(cfg('SURFACE_LIMIT_RATIO')) # relative to equal-weight baseline, exploratory
OFFTARGET_PENALTY = float(cfg('OFFTARGET_PENALTY'))
DE_POPSIZE = int(cfg('DE_POPSIZE'))
DE_MAXITER = int(cfg('DE_MAXITER'))
DE_TOL = float(cfg('DE_TOL'))
if TARGET_MNI.shape != (3,) or BASELINE.shape != (3,):
    raise ValueError('TARGET_MNI and BASELINE_WEIGHTS need exactly 3 comma-separated values')
print('Config:', ENV_FILE)
print('Input meshes:', *(f'{k}: {v}' for k,v in MESH_PATHS.items()), sep='\n  ')
print('Output:', OUTPUT)
print('FEM reruns: DISABLED')

paths = {}
for name, path in MESH_PATHS.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: mesh file not found: {path}\nEdit {name}_MSH or INPUT_DIR in {ENV_FILE} and rerun.")
    if path.suffix.lower() != '.msh':
        raise ValueError(f"{name}: expected a .msh file: {path}")
    paths[name] = path
    print(f"PASS {name}: {path}")

meshes={k:mesh_io.read_msh(str(v)) for k,v in paths.items()}
ref=meshes['P1']
xyz=np.asarray(ref.nodes.node_coord,float)
conn=np.asarray(ref.elm.node_number_list)
types=np.asarray(ref.elm.elm_type)
tags=np.asarray(ref.elm.tag1)
fields=[]
for name in NAMES:
    mesh=meshes[name]
    if not np.array_equal(np.asarray(mesh.elm.elm_type),types) or not np.array_equal(np.asarray(mesh.elm.tag1),tags):
        raise RuntimeError(f'{name}: element types/tags differ')
    if not np.array_equal(np.asarray(mesh.elm.node_number_list),conn):
        raise RuntimeError(f'{name}: connectivity differs')
    if not np.allclose(np.asarray(mesh.nodes.node_coord,float),xyz,atol=1e-6,rtol=0):
        raise RuntimeError(f'{name}: node coordinates differ')
    matches=[f for f in mesh.elmdata if f.field_name=='E']
    if len(matches)!=1: raise RuntimeError(f'{name}: expected one element E field, found {len(matches)}')
    e=np.asarray(matches[0].value,float)
    if e.shape!=(len(types),3) or not np.all(np.isfinite(e)):
        raise RuntimeError(f'{name}: invalid E shape/values {e.shape}')
    fields.append(e)
E=np.stack(fields,axis=0)
print('PASS: aligned E basis', E.shape, 'V/m per Rafael reference excitation')

target_subject=np.asarray(transformations.mni2subject_coords(TARGET_MNI.reshape(1,3),str(HEAD_DIR)),float).reshape(3)
tet=(types==4)
idx=np.flatnonzero(tet)
vertices=xyz[conn[idx,:4]-1]
centers=vertices.mean(axis=1)
vol=np.abs(np.einsum('ij,ij->i',vertices[:,1]-vertices[:,0],np.cross(vertices[:,2]-vertices[:,0],vertices[:,3]-vertices[:,0])))/6
if not np.all(vol>0): raise RuntimeError('Nonpositive tetrahedral volumes')
mask_target=(tags[idx]==GM_TAG)&(np.linalg.norm(centers-target_subject,axis=1)<=ROI_RADIUS_MM)
mask_other=(tags[idx]==GM_TAG)&~mask_target
mask_surface=(tags[idx]==SCALP_TAG)
for label,mask in [('target',mask_target),('other gray matter',mask_other),('scalp proxy',mask_surface)]:
    if not mask.any():raise RuntimeError(f'Empty region: {label}; check tissue tags')
    print(label, int(mask.sum()),'tetrahedra')
E_tet=E[:,idx,:]
print('Target subject coordinates:',target_subject)

def field_magnitude(a):
    a=np.asarray(a,float)
    if a.shape!=(3,): raise ValueError('Expected 3 coefficients')
    return np.linalg.norm(np.einsum('k,kij->ij',a,E_tet),axis=1)

def roi_mean(magnitude,mask):
    return float(np.average(magnitude[mask],weights=vol[mask]))

def metrics(a):
    mag=field_magnitude(a)
    return {'target':roi_mean(mag,mask_target), 'surface':roi_mean(mag,mask_surface), 'offtarget':roi_mean(mag,mask_other)}

for k in range(3):
    a=np.eye(3)[k]
    print(NAMES[k],metrics(a))

base = metrics(BASELINE)
SURFACE_CAP = base['surface'] * SURFACE_LIMIT_RATIO
print('Equal-weight baseline:', base)
print('Exploratory surface cap:', SURFACE_CAP)

def unpack(x):
    """Two independent variables; exact total allocation MAX_SUM."""
    return np.array([x[0], x[1], MAX_SUM - x[0] - x[1]])

def feasible(a):
    a = np.asarray(a, dtype=float)
    return (a.shape == (3,) and np.all(a >= COIL_MIN - 1e-9)
            and np.all(a <= COIL_MAX + 1e-9)
            and abs(a.sum() - MAX_SUM) <= 1e-8
            and metrics(a)['surface'] <= SURFACE_CAP + 1e-9)

def objective(x):
    a = unpack(x)
    if np.any(a < COIL_MIN) or np.any(a > COIL_MAX):
        return 1e4 + 1e3 * np.sum(np.maximum(COIL_MIN-a, 0) + np.maximum(a-COIL_MAX, 0))
    m = metrics(a)
    # Favor the target while penalizing off-target exposure and surface-cap violations.
    return (-m['target']/base['target']
            + OFFTARGET_PENALTY*m['offtarget']/base['offtarget']
            + 1e3*max(0.,m['surface']/SURFACE_CAP-1.)**2)


res=differential_evolution(objective,[(COIL_MIN,COIL_MAX)]*2,seed=SEED,popsize=DE_POPSIZE,maxiter=DE_MAXITER,tol=DE_TOL,polish=True)
optimal=unpack(res.x)
if not feasible(optimal): raise RuntimeError(f'Optimizer returned infeasible result: {optimal}, {metrics(optimal)}')
rows=[]
for label,a in [('equal_budget',BASELINE),('optimized',optimal),('P1_only',np.array([1.,0,0])),('P2_only',np.array([0,1.,0])),('P3_only',np.array([0,0,1.]))]:
    rows.append({'candidate':label,**{f'a{i+1}':float(a[i]) for i in range(3)},'sum_a':float(sum(a)),**metrics(a)})
comparison=pd.DataFrame(rows)
print('Optimization:',res.message)
print(comparison.to_string(index=False))


# Independent single-coil limiting-case validation
# No FEM reruns. Original datasets remain unchanged.

regions = {
    "target": mask_target,
    "surface": mask_surface,
    "offtarget": mask_other
}

# Extract tetrahedral fields from the full FEM basis.
E_tet = E[:, tet, :]

for k, name in enumerate(NAMES):
    weights = np.eye(3)[k]
    calculated = metrics(weights)

    print(f"\n{name} validation:")

    for region, mask in regions.items():

        # Region masks already index the tetrahedral subset.
        original_E = np.linalg.norm(
            E_tet[k, mask, :], axis=1
        )

        reference = np.average(
            original_E,
            weights=vol[mask]
        )

        error = abs(calculated[region] - reference)

        print(
            f"  {region}: calculated={calculated[region]:.8f}, "
            f"reference={reference:.8f}, "
            f"error={error:.2e}"
        )

        assert np.isclose(
            calculated[region],
            reference,
            rtol=1e-10,
            atol=1e-12
        )

print("\nPASS: All three single-coil limiting cases.")


# Final superposition identity checks
# Uses existing FEM data only. No new simulations.

zero = metrics(np.array([0.0, 0.0, 0.0]))
full = metrics(np.array([1.0, 0.0, 0.0]))
half = metrics(np.array([0.5, 0.0, 0.0]))

for region in ("target", "surface", "offtarget"):
    assert np.isclose(zero[region], 0.0, atol=1e-12)
    assert np.isclose(
        half[region], 0.5 * full[region],
        rtol=1e-10, atol=1e-12
    )
    print(
        f"{region}: zero={zero[region]:.8f}, "
        f"half={half[region]:.8f}, "
        f"expected={0.5 * full[region]:.8f}"
    )

print("\nPASS: Zero-field and linear-scaling identities.")

OUTPUT.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT / f'{OUTPUT_STEM}.csv'
png_path = OUTPUT / f'{OUTPUT_STEM}.png'
json_path = OUTPUT / f'{OUTPUT_STEM}.json'
comparison.to_csv(csv_path, index=False)
labels = ['equal_budget', 'optimized', 'P2_only']
plot_data = comparison.set_index('candidate').loc[labels, ['target', 'surface', 'offtarget']]
ax = plot_data.plot(kind='bar', figsize=(9, 4.5), rot=0)
ax.set_ylabel('Volume-weighted mean |E| (V/m)')
ax.set_xlabel('Coil configuration')
ax.set_title('Three-coil superposition: baseline, optimized and P2-only')
ax.legend(title='Region')
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(png_path, dpi=180)
record = {
    'config_file': str(ENV_FILE),
    'input_meshes': {k: str(v) for k, v in paths.items()},
    'target_mni_mm': TARGET_MNI.tolist(),
    'target_subject_mm': target_subject.tolist(),
    'roi_radius_mm': ROI_RADIUS_MM,
    'gm_tag': GM_TAG,
    'scalp_tag': SCALP_TAG,
    'reference_didt_A_per_s': REFERENCE_DIDT,
    'coil_weight_bounds': [COIL_MIN, COIL_MAX],
    'baseline_weights': BASELINE.tolist(),
    'optimized_weights': optimal.tolist(),
    'normalized_three_coil_budget': MAX_SUM,
    'surface_cap_ratio_to_baseline': SURFACE_LIMIT_RATIO,
    'offtarget_penalty': OFFTARGET_PENALTY,
    'optimizer_seed': SEED,
    'optimizer_settings': {'popsize': DE_POPSIZE, 'maxiter': DE_MAXITER, 'tol': DE_TOL},
    'optimizer_success': bool(res.success),
    'optimizer_feasible': bool(feasible(optimal)),
    'source': 'Rafael three-position FEM datasets; Victor reference execution',
    'constraints_exploratory': True,
    'single_coil_comparisons_not_budget_matched': True
}
json_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
print('Saved CSV:', csv_path)
print('Saved graph:', png_path)
print('Saved JSON:', json_path)
plt.show()