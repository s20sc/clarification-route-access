# Clarification instructions and route dependent access

Offline analysis and selected controller sources for **Clarification instructions and route-dependent access in simulated household manipulation**.

The study compares three static instruction conditions at two fixed household stacking windows. The clarification instruction changes questioning under the tested prompts. The route contrast fails the cross-window criterion, and the same-information rule selects the same dispatched route. The code does not establish a personality-growth mechanism or a population task-success advantage.

## Reproduce the archived comparisons

Download the data ZIP files listed in `data-sha256.json` from the companion Zenodo dataset (reserved DOI: `10.5281/zenodo.24002912`). Before publication, the corresponding authors provide a draft preview link for peer review; the reserved DOI alone does not yet provide access. With Python 3.9 or newer, run:

```sh
python3 -I -B reproduce.py --data-dir /path/to/downloaded/files --output-dir /path/to/results
```

The quantitative and controller archives are required. Add the observation and additional-original-video archives to verify all twelve original recordings. The display archive is optional. The command reports any checks not run. It requires no Python packages, network, model service or simulator.

The command checks download hashes, verifies that the source mirror matches the archived programs, and executes the frozen programs in a temporary extraction directory. One source heading and one docstring have reader-facing wording; their executable statements are unchanged. Original evidence and file hashes are retained in the dataset. `src/quantitative` holds the analysis and decision logic, `src/controllers` holds controller excerpts and verification, and `src/observations` and `src/additional_movies` hold recording integrity checks. Historical experiment directory numbers identify inputs in the dataset.

This is an offline reanalysis, not a portable fresh simulator run. Fresh inference and physical execution need separately obtained model weights, serving software, simulator assets and a suitable execution environment. The recorded coding reports retain unknown labels and missing checkpoint identifiers. The evidence packages overlap; their input counts are not independent sample sizes.

## Authors and reuse

Xue Qin, Simin Luan, Cong Yang and Zhijun Li. Correspondence: cong.yang@suda.edu.cn; lizhijun_os@hit.edu.cn.

Third-party software and assets retain their original terms. Availability of the source excerpts does not grant additional rights in third-party software or assets.
