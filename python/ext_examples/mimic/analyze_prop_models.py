from mimic.models import ImageAutoEncoder, LSTM, DenseProp
from mimic.dataset import AutoRegressiveDataset
from mimic.trainer import TrainCache
from mimic.predictor import evaluate_command_prediction_error
from mimic.scripts.train_propagator import prepare_trained_image_chunk

project_name = 'kuka_reaching'
n_intact = 5
chunk = prepare_trained_image_chunk(project_name)
chunk_intact, _ = chunk.split(n_intact)
dataset = AutoRegressiveDataset.from_chunk(chunk_intact)

tcache_ae = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
tcache_lstm = TrainCache[LSTM].load(project_name, LSTM)
tcache_dense = TrainCache[DenseProp].load(project_name, DenseProp)

val_lstm = evaluate_command_prediction_error(tcache_ae.best_model, tcache_lstm.best_model, dataset)
print(val_lstm)

val_dense = evaluate_command_prediction_error(tcache_ae.best_model, tcache_dense.best_model, dataset)
print(val_dense)

