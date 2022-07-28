import numpy as np
from dataclasses import dataclass
from scipy import ndarray

from sklearn.decomposition import PCA
from mohou.encoding_rule import EncodingRule

from typing import List


@dataclass
class PCAWrapper:
    pca: PCA

    @classmethod
    def from_vectors(cls, vectors: List[np.ndarray], n_dim: int):
        mat = np.array(vectors)
        pca = PCA(n_components=n_dim)
        pca.fit(mat)
        return cls(pca)

    def transform(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        out = self.pca.transform(np.array(vectors))
        return list(out)



if __name__ == "__main__":
    from mohou.file import get_project_path
    from mohou.types import EpisodeBundle, GripperState
    from mohou.default import create_default_encoding_rule
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    project_name = "pr2_tidyup_dish"
    project_path = get_project_path(project_name)
    bundle = EpisodeBundle.load(project_path)

    encoding_rule = create_default_encoding_rule(project_path)
    vector_seq_list = encoding_rule.apply_to_episode_bundle(bundle)
    print("converted")

    vector_list = []
    for vector_seq in vector_seq_list:
        vector_list.extend(vector_seq)

    pca = PCAWrapper.from_vectors(vector_list, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(10):
        vector_seq = vector_seq_list[i]
        comp_vectors = pca.transform(vector_seq)
        M = np.array(comp_vectors)
        ax.plot(M[:, 0], M[:, 1], M[:, 2])

    plt.show()
