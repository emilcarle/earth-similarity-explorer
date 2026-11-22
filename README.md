# AlphaEarth Similarity Explorer

> Work in progress.

A web page for exploring similarity in Google's AlphaEarth satellite embeddings. Click a place in the study area, and every place whose embedding is similar enough gets highlighted; a slider sets the similarity threshold. A 3D view shows the same pixels as a point cloud on the embeddings' first three principal components.

## License

The code is released under the [MIT License](LICENSE). The data comes from third parties and keeps their licenses:

| Data | In this repository | License | Attribution |
| --- | --- | --- | --- |
| Study-area boundaries | `geometries/` | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) | © OpenStreetMap contributors |
| Satellite embeddings and data derived from them | read through Earth Engine | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind. Brown et al. (2025), [arXiv:2507.22291](https://arxiv.org/abs/2507.22291) |
