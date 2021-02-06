using Altro
using TrajectoryOptimization
using StaticArrays, LinearAlgebra
using RobotDynamics

struct PointModel <: AbstractModel end
function RobotDynamics.dynamics(model::PointModel, x, u)
    return u
end
size(model::PointModel) = (2, 2)
RobotDynamics.state_dim(model::PointModel) = 2
RobotDynamics.control_dim(model::PointModel) = 2

model = PointModel()
n,m = size(model)

N = 21
tf = 5.
dt = tf/(N-1)

x0 = @SVector zeros(n)
xf = @SVector ones(n)
u0 = @SVector fill(0.0,m)
U0 = [u0 for k = 1:N-1]

Q = 0.0*Diagonal(@SVector ones(n))
Qf = 0.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N)

conSet = ConstraintList(n,m,N)
goal = GoalConstraint(xf)

struct MyConstraint <: TrajectoryOptimization.StateConstraint end
function TrajectoryOptimization.evaluate(con::MyConstraint, X::StaticVector)
    cost = (X[1] - 0.5)^2 + (X[2] - 0.5)^2 - 0.4^2
    return - cost
end
function TrajectoryOptimization.jacobian!(jac, con::MyConstraint, X::SVector)
    jac[1, 1] = - 2 * X[1]
    jac[1, 2] = - 2 * X[2]
	return true
end
@inline TrajectoryOptimization.sense(::MyConstraint) = Inequality()
@inline TrajectoryOptimization.state_dim(::MyConstraint) = 2
@inline TrajectoryOptimization.length(::MyConstraint) = 1

circle = MyConstraint()
add_constraint!(conSet, goal, N)

prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(prob, U0)

opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)
altro = ALTROSolver(prob, opts)
@time solve!(altro);
X = states(altro)
U = controls(altro)

add_constraint!(conSet, circle, 1:N)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(prob, U)

opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)
altro = ALTROSolver(prob, opts)
@time solve!(altro);
X = states(altro)
U = controls(altro)
