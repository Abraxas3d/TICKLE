# TICKLE
## Ten-GHz Intensity & Cloud Kinematic Learning Experiment

_by Jayfeather3d, Abraxas3d, Skunkworks_

TICKLE is a cloud bounce prediction model.

**TICKLE is a "Nowcast" vs. "Forecast".** Sites and programs such as rainscatter.com show you where reflectivity is right now. In other words, it is a real-time map. TICKLE's founding premise, and the reason GOES imagery is the input rather than radar, is lead time. We want to predict the mirror before it's a mirror, from the cloud-top evolution satellite sees earlier than radar sees rain. "Is it open now" versus "will it open in 30–60 minutes" is a categorical difference in usefulness to someone deciding whether to drive to a summit or choose a particular operating style. 

**TICKLE gives path-specific viability vs. a reflectivity map.** This is the geometry contribution, and the Phase 1 finding is the proof that the volumetric geometry issues are non-trivial. A dBZ map tells you a storm exists. It does not tell you whether that storm sits in the common volume of your specific path to another station. Sites such as rainscatter.com show weather. TICKLE answers "can I, from here, work there." That's the terrain-geometry layer nobody's published for amateur cloud bounce paths as far as any of us know.

**TICKLE uses learned link-viability vs. a reflectivity threshold.** This is the eventual machine learning payoff, and it's where we go beyond simply returning the dBZ. A threshold ("40 dBZ = workable") is a rule of thumb. A model trained on real QSO outcomes against a full feature set, including echo-top height, mirror volume in the common region, cloud-top cooling rate, band differences, path geometry, learns the actual decision surface. This includes the interactions a single threshold can't capture. For example, in Phase 1, the shallow-violent 2021 morning cells vs. the deep 2023 towers were the same dBZ with different physics.

TICKLE aims to be a path-specific, terrain-aware, lead-time predictor validated against contest outcomes, logged contacts, and beacon work. Same raw ingredient (radar is one of the inputs), but a different product. We are not competing with sites and programs like rainscatter.com. We are building something that these types of maps would be one input to. 
