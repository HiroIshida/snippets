import tqdm
import logging
from hifuku.script_utils import load_library, load_sampler_history
from hifuku.domain import FetchConwayJailInsert
from hifuku.core import ActiveSamplerHistory
from hifuku.pool import PredicatedTaskPool, TaskPool
from rpbench.interface import BytesArrayWrap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logger.txt'),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

domain = FetchConwayJailInsert
tt = domain.task_type
history: ActiveSamplerHistory = load_sampler_history(domain, postfix="0.1")
lib = load_library(domain, "cuda", postfix="0.1")
init_solutions = lib.init_solutions
# lib.jit_compile(True)
solver = domain.solver_type.init(domain.solver_config)
n_task = 3000
# taskset_presample = BytesArrayWrap([tt.sample().to_task_param() for _ in range(n_task)])
sampler = domain.get_multiprocess_batch_sampler(8)
pool = TaskPool(tt).as_predicated()
taskset_presample = sampler.sample_batch(n_task, pool)
logger.info("start sampling")
solver = domain.get_multiprocess_batch_solver(8)
logger.info("start solving")

for it in tqdm.tqdm(range(len(history.biases_history))):
    agg = history.aggregate_list[it]
    rate_recored = sum(agg.reals < agg.threshold) / len(agg.reals)
    traj_guess = lib.init_solutions[it]
    results = solver.solve_batch(taskset_presample, [traj_guess] * n_task) 
    success_count = sum([res.traj is not None for res in results])
    rate_here = success_count / n_task
    logger.info(f"recorded rate = {rate_recored}, here rate = {rate_here}")
