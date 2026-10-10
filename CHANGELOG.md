# Changelog

## [0.0.8](https://github.com/jejjohnson/geonnax/compare/v0.0.7...v0.0.8) (2026-10-10)


### Features

* **basis:** add icosphere triangle mesh (GN3) ([#58](https://github.com/jejjohnson/geonnax/issues/58)) ([3f382a4](https://github.com/jejjohnson/geonnax/commit/3f382a4e57afe6a090922960c7d379c92eae8eb4))
* **basis:** add Legendre and Gauss–Legendre primitives (GN1) ([#56](https://github.com/jejjohnson/geonnax/issues/56)) ([5225bbf](https://github.com/jejjohnson/geonnax/commit/5225bbf6f413879aa6c3ab6d171fbc54da3e7fd4))
* **basis:** add Neumann/periodic box bases and curlfree_basis (GN8) ([#64](https://github.com/jejjohnson/geonnax/issues/64)) ([4687645](https://github.com/jejjohnson/geonnax/commit/4687645930b863e920f732d9837cfc1765721b9b))
* **basis:** add Slepian bases for general regions and polar gaps (GN6) ([#62](https://github.com/jejjohnson/geonnax/issues/62)) ([32cc758](https://github.com/jejjohnson/geonnax/commit/32cc758fa990a22c6a95524cc999cdcd5f86c020))
* **basis:** add sphere point sets and quadrature grids (GN2) ([#57](https://github.com/jejjohnson/geonnax/issues/57)) ([348f590](https://github.com/jejjohnson/geonnax/commit/348f590c860c5aed76e3653d7dbf2b55885f6670))
* **basis:** add spherical needlets (GN7) ([#63](https://github.com/jejjohnson/geonnax/issues/63)) ([9e055c6](https://github.com/jejjohnson/geonnax/commit/9e055c6c481c78714da45516ea2ce90ef85f49c2))
* **basis:** add spherical-shell basis (GN9) ([#65](https://github.com/jejjohnson/geonnax/issues/65)) ([2b420c9](https://github.com/jejjohnson/geonnax/commit/2b420c95d53ae14c5f1e02f74b37d5c3023eaa66))
* **basis:** add vector spherical harmonics (GN5) ([#61](https://github.com/jejjohnson/geonnax/issues/61)) ([871face](https://github.com/jejjohnson/geonnax/commit/871facedfcfe5be63cc7c39a587f617d56d9ff81))
* **geo:** add coordinate frames — WGS84/ECEF, ENU, tangent vectors, tangent plane, rotation to pole (GN4) ([#60](https://github.com/jejjohnson/geonnax/issues/60)) ([2084677](https://github.com/jejjohnson/geonnax/commit/20846771693e6e8356ecad751c8b702b1ba7769f))

## [0.0.7](https://github.com/jejjohnson/geonnax/compare/v0.0.6...v0.0.7) (2026-08-21)


### Features

* deterministic NN building blocks for the pyrox stack ([#30](https://github.com/jejjohnson/geonnax/issues/30)) ([#40](https://github.com/jejjohnson/geonnax/issues/40)) ([de87394](https://github.com/jejjohnson/geonnax/commit/de87394a2fa60d4462818b470ac8d900ed359d15))

## [0.0.6](https://github.com/jejjohnson/geonnax/compare/v0.0.5...v0.0.6) (2026-08-20)


### Features

* **encoders:** add GeoContextEncoder for lat/lon/time/covariate context vectors ([#38](https://github.com/jejjohnson/geonnax/issues/38)) ([359be95](https://github.com/jejjohnson/geonnax/commit/359be957332b9841efa4e1335c7aff2727903e26))

## [0.0.5](https://github.com/jejjohnson/geonnax/compare/v0.0.4...v0.0.5) (2026-06-08)


### Features

* **basis:** public basis surface + EOF, divergence-free, geodesic-RBF & wavelet bases ([#25](https://github.com/jejjohnson/geonnax/issues/25)) ([9f160d3](https://github.com/jejjohnson/geonnax/commit/9f160d37e455530291e964421271d8079372fd5b))

## [0.0.4](https://github.com/jejjohnson/geonnax/compare/v0.0.3...v0.0.4) (2026-06-07)


### Features

* **basis:** localized RBF, overcomplete Gabor frame, and Gaussian time window ([#22](https://github.com/jejjohnson/geonnax/issues/22)) ([e3a0d57](https://github.com/jejjohnson/geonnax/commit/e3a0d57bf3a3116065970e8908d220259b2aa9c0))
* **mswt:** add wavelet-domain attention and the Multi-Scale Wavelet Transformer ([#21](https://github.com/jejjohnson/geonnax/issues/21)) ([1431af2](https://github.com/jejjohnson/geonnax/commit/1431af29ae700cf03ca67f13a9979a3209f59c4b))
* **spherical-wavelets:** scale-discretised wavelet transform on the sphere ([#16](https://github.com/jejjohnson/geonnax/issues/16)) ([da7616f](https://github.com/jejjohnson/geonnax/commit/da7616f2f79dc6967138e1f527fdee5bffac46d5))
* **unet:** thread optional dropout through U-Net blocks for MC-dropout ([#20](https://github.com/jejjohnson/geonnax/issues/20)) ([3675051](https://github.com/jejjohnson/geonnax/commit/3675051f6d34c02712978890e395e5a4a843856c))
* **wno:** discrete wavelet transform layers and Wavelet Neural Operator ([#15](https://github.com/jejjohnson/geonnax/issues/15)) ([0fca52e](https://github.com/jejjohnson/geonnax/commit/0fca52e0bb961c73ffa0a47a5b8efe2d291827f1))

## [0.0.3](https://github.com/jejjohnson/geonnax/compare/v0.0.2...v0.0.3) (2026-06-01)


### Features

* **fno:** add Fourier and Spherical Neural Operators ([#9](https://github.com/jejjohnson/geonnax/issues/9)) ([39f92bf](https://github.com/jejjohnson/geonnax/commit/39f92bf60969d7ec8114612bcdd32b80dcea188a))
* **unet:** add dimension-flexible U-Net and reusable conv layers ([#7](https://github.com/jejjohnson/geonnax/issues/7)) ([b3eeffb](https://github.com/jejjohnson/geonnax/commit/b3eeffbe27bf564b54f9b94dbe1630e30d85b151))

## [0.0.2](https://github.com/jejjohnson/geonnax/compare/v0.0.1...v0.0.2) (2026-06-01)


### Features

* full pure-Equinox neural-network zoo (geonnax v0.0.1) ([#4](https://github.com/jejjohnson/geonnax/issues/4)) ([696c122](https://github.com/jejjohnson/geonnax/commit/696c122e7045d7689895f319e024061e06f10046))

## 0.0.1 (2026-06-01)


### Features

* full pure-Equinox neural-network zoo (geonnax v0.0.1) ([#4](https://github.com/jejjohnson/geonnax/issues/4)) ([696c122](https://github.com/jejjohnson/geonnax/commit/696c122e7045d7689895f319e024061e06f10046))
