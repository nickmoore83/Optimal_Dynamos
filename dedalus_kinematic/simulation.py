
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx = Ly = Lz = 2*np.pi
Nx = Ny = Nz = 64
N0 = 4
w = 0.5
Rm = 5
update_u = True
dealias = 3/2
timestepper = d3.RK222
timestep = 5e-3
stop_sim_time = 5
dtype = np.float64
seed = 4

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
B0 = dist.VectorField(coords, name='B0', bases=(xbasis,ybasis,zbasis))
B = dist.VectorField(coords, name='B', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
p1 = dist.Field(name='p1', bases=(xbasis,ybasis,zbasis))
p2 = dist.Field(name='p2', bases=(xbasis,ybasis,zbasis))
p3 = dist.Field(name='p3', bases=(xbasis,ybasis,zbasis))
c1 = dist.Field(name='c1')
c2 = dist.Field(name='c2')
c3 = dist.Field(name='c3')

# Substitutions
ω = d3.curl(u)
j = d3.curl(B)
#E = (w/2)*d3.ave(u@u) + ((1-w)/2)*d3.ave(ω@ω)
energy = 0.5*d3.ave(u@u)
enstrophy = 0.5*d3.ave(ω@ω)
u_norm = w*energy + (1-w)*enstrophy
M = d3.ave(B@B)

# Velocity LBVP
u_lbvp = d3.LBVP(variables=[u, p1, c1], namespace=locals())
u_lbvp.add_equation("w*u - (1-w)*lap(u) + grad(p1) = - cross(j, B)")
u_lbvp.add_equation("div(u) + c1 = 0")
u_lbvp.add_equation("integ(p1) = 0")
u_lbvp = u_lbvp.build_solver()

def solve_u():
    u_lbvp.solve()
    u_norm0 = u_norm.evaluate().allgather_data()[0,0,0]
    u['c'] /= u_norm0**0.5
    u_norm.evaluate()
    return u

# Divergence cleaning
B_div = d3.LBVP(variables=[B, p2, c2], namespace=locals())
B_div.add_equation("B + grad(p2) = B0")
B_div.add_equation("div(B) + c2 = 0")
B_div.add_equation("integ(p2) = 0")
B_div = B_div.build_solver()

# Magnetic IVP
B_ivp = d3.IVP(variables=[B, p3, c3], namespace=locals())
B_ivp.add_equation("dt(B) - (1/Rm)*lap(B) + grad(p3) = curl(cross(u, B))")
B_ivp.add_equation("div(B) + c3 = 0")
B_ivp.add_equation("integ(p3) = 0")
B_solver = B_ivp.build_solver(timestepper)
B_solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = B_solver.evaluator.add_file_handler('snapshots', iter=10, max_writes=10)
snapshots.add_task(B, name='B')
snapshots.add_task(u, name='u')
snapshots.add_task(ω, name='ω')
scalars = B_solver.evaluator.add_file_handler('scalars', iter=1)
scalars.add_task(energy, name='energy')
scalars.add_task(enstrophy, name='enstrophy')
scalars.add_task(u_norm, name='u_norm')
scalars.add_task(M, name='M')

# Flow properties
flow = d3.GlobalFlowProperty(B_solver, cadence=10)
flow.add_property(energy, name='energy')
flow.add_property(enstrophy, name='enstrophy')
flow.add_property(u_norm, name='u_norm')
flow.add_property(M, name='M')

# Initial conditions
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
B0.fill_random(seed=seed)
B0.low_pass_filter(scales=(N0/Nx, N0/Ny, N0/Nz))
B_div.solve()
solve_u()

# Main loop
try:
    logger.info('Starting main loop')
    while B_solver.proceed:
        B_solver.step(timestep)
        if update_u:
            solve_u()
        if (B_solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, u_norm=%.2e, M=%.2e' %(B_solver.iteration, B_solver.sim_time, timestep, flow.max('u_norm'), flow.max('M')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    B_solver.log_stats()

