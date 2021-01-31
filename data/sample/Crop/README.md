# Crop

The new generation Earth Observation (EO) satellites have been imaging the Earth frequently, completely and at high resolution. This introduces unprecedented opportunities to monitor the dynamics of any regions on our planet over time and revealing the constant flux that underpins the bigger picture of our World. 

This data set is a subset of a bigger data set [1], which originally comes from 46 geometrically and radiometrically corrected images taken by FORMOSAT-2 satellite. These images are corrected such that every pixel corresponds to the same geographic area on Earth. Each pixel represents an area of 64 square meter; with 1 million pixels per image, this results in an area of 64 square kilometer per image. Each geographic area (a pixel) forms a time series of length of 46, showing the temporal evolution of that area.

There are 24 classes corresponding to what the land covers. 

Class label - class name in French: English translation

- Class 1 - mais: corn
- Class 2 - ble: wheat
- Class 3 - bati dense: dense building
- Class 4 - bati indu: built indu
- Class 5 - bati diffus: diffuse building
- Class 6 - prairie temporaire: temporary meadow
- Class 7 - feuillus: hardwood
- Class 8 - friche: wasteland
- Class 9 - jachere: jachere
- Class 10 - soja: soy
- Class 11 - eau: water
- Class 12 - pre: pre
- Class 13 - resineux: softwood
- Class 14 - tournesol: sunflower
- Class 15 - sorgho: sorghum
- Class 16 - eucalyptus: eucalyptus
- Class 17 - colza: rapeseed
- Class 18 - mais ensillage: but drilling
- Class 19 - orge: barley
- Class 20 - pois: peas
- Class 21 - peupliers: poplars
- Class 22 - surface minerale: mineral surface
- Class 23 - graviere: gravel
- Class 24 - lac: lake

Train size: 7200

Test size: 16800

Number of classes: 24

Missing value: No

Time series length: 46

There is nothing to infer from the order of examples in the train and test set.

Data created by: C.W. Tan, G.I. Webb and F. Petitjean (see [1], [2]). Data edited by Hoang Anh Dau.

[1] Tan, Chang Wei, Geoffrey I. Webb, and François Petitjean. "Indexing and classifying gigabytes of time series under time warping." Proceedings of the 2017 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2017.

[2] http://bit.ly/SDM2017
