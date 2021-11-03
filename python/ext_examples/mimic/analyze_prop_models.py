from mimic.models import ImageAutoEncoder, LSTM, DenseProp, DeprecatedDenseProp
from mimic.dataset import AutoRegressiveDataset, FirstOrderARDataset
from mimic.trainer import TrainCache
from mimic.predictor import evaluate_command_prediction_error
from mimic.scripts.train_propagator import prepare_trained_image_chunk

project_name = 'kuka_reaching'
n_intact = 5
chunk = prepare_trained_image_chunk(project_name)
chunk_intact, _ = chunk.split(n_intact)

tcache_ae = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
tcache_lstm = TrainCache[LSTM].load(project_name, LSTM)
tcache_dense = TrainCache[DenseProp].load(project_name, DenseProp)
tcache_depre = TrainCache[DeprecatedDenseProp].load(project_name, DeprecatedDenseProp)

val_lstm = evaluate_command_prediction_error(tcache_ae.best_model, tcache_lstm.best_model, chunk_intact)
print(val_lstm)

val_dense = evaluate_command_prediction_error(tcache_ae.best_model, tcache_dense.best_model, chunk_intact)
print(val_dense)

val_dense = evaluate_command_prediction_error(tcache_ae.best_model, tcache_depre.best_model, chunk_intact)
print(val_dense)
