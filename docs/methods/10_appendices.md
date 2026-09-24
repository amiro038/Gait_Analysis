# Appendix A. Every setting

Every number that shapes a result is a setting at the top of a script. The tables below are generated from the scripts themselves, so they always match the code.

## A.1 detect_gait_events.py (CONFIG)

{{params: detect_gait_events.py}}

: Table A.1. Settings of the event detection.

## A.2 gait_analysis.py (§0 SETTINGS)

{{params: gait_analysis.py}}

: Table A.2. Settings of the gait analysis. `TRUNK_SOURCES` (§16) and `LDS_STATE_SPACES` (§18) are lists, and are described in their sections.

# Appendix B. What a load is expected to do

These are the directions reported in the load-carriage literature for healthy adults. They are **expected directions, not ranges**: use them to judge whether a value outside the healthy range is the effect of the load or a problem. Each participant's own unloaded condition is the best reference.

| Metric | Expected change with load | Source |
|:--|:--|:--|
| F1, F2, impulses per body weight (`_bw`) | increase, roughly in proportion to the total weight | Birrell et al. 2007 |
| the same per total weight (`_tw`) | little change | Birrell et al. 2007 |
| loading rate | increases | Birrell et al. 2007 |
| double support, stance % | increase | Kinoshita 1985; Knapik et al. 2004 |
| cadence / stride length | cadence up and stride shorter, slightly (fixed here by the metronome and the belt) | Knapik et al. 2004 |
| trunk forward lean | increases, most with backpacks | Kinoshita 1985; Attwells et al. 2006 |
| knee flexion in weight acceptance, knee ROM | increases | Kinoshita 1985; Attwells et al. 2006 |
| CoM work (collision, push-off) | expected to increase with the mass redirected at each step | Donelan et al. 2002 (model) |
| ML margin of stability, step width | mixed evidence; depends on load placement | – |
| MFC, variability, entropy, LDS | mixed evidence; fatigue under load may lower MFC | Qu & Yeo 2011 |

: Table B.1. Expected direction of the effect of load carriage.

# Appendix C. Where each section is in the code

Every section of `gait_analysis.py`, in order, with the functions it defines. The table is generated from the script. In Spyder, the outline pane (View → Panes → Outline) lists the same sections, so you can jump to one.

{{codemap}}

: Table C.1. The sections and functions of gait_analysis.py.

**The run, one trial** (`analyse_trial()`):

1. `load_trial()` carries out §5.1 → §5.7;
2. `spatiotemporal()` (§6), `posture()` (§7), `kinetics()` (§8–9), `margin_of_stability()` (§10) and `foot_clearance()` (§11) add their columns to the per-step table;
3. `summarise()` computes §12–18 and returns the trial's long table;
4. the files of §19 are written.

`main()` does this for every trial and then calls `build_tables()`.

# Appendix D. References

Attwells, R.L., Birrell, S.A., Hooper, R.H., Mansfield, N.J. (2006). Influence of carrying heavy loads on soldiers' posture, movements and gait. *Ergonomics* 49, 1527–1537.

Barrett, R.S., Mills, P.M., Begg, R.K. (2010). A systematic review of the effect of ageing and falls history on minimum foot clearance characteristics during level walking. *Gait & Posture* 32, 429–435.

Begg, R., Best, R., Dell'Oro, L., Taylor, S. (2007). Minimum foot clearance during walking: reliability and variability in young and elderly. *Gait & Posture* 25, 191–198.

Birrell, S.A., Hooper, R.H., Haslam, R.A. (2007). The effect of military load carriage on ground reaction forces. *Gait & Posture* 26, 611–614.

Bruijn, S.M., Meijer, O.G., Beek, P.J., van Dieën, J.H. (2013). Assessing the stability of human locomotion: a review of current measures. *Journal of the Royal Society Interface* 10, 20120999.

Costa, M., Goldberger, A.L., Peng, C.-K. (2002). Multiscale entropy analysis of complex physiologic time series. *Physical Review Letters* 89, 068102.

Damouras, S., Chang, M.D., Sejdić, E., Chau, T. (2010). An empirical examination of detrended fluctuation analysis for gait data. *Gait & Posture* 31, 336–340.

Davies, R.B., Harte, D.S. (1987). Tests for Hurst effect. *Biometrika* 74, 95–101.

Dingwell, J.B., Cusumano, J.P. (2000). Nonlinear time series analysis of normal and pathological human walking. *Chaos* 10, 848–863.

Dingwell, J.B., Cusumano, J.P. (2015). Identifying stride-to-stride control strategies in human treadmill walking. *PLoS ONE* 10, e0124879.

Dingwell, J.B., John, J., Cusumano, J.P. (2010). Do humans optimally exploit redundancy to control step variability in walking? *PLoS Computational Biology* 6, e1000856.

Donelan, J.M., Kram, R., Kuo, A.D. (2002). Simultaneous positive and negative external mechanical work in human walking. *Journal of Biomechanics* 35, 117–124.

Geweke, J., Porter-Hudak, S. (1983). The estimation and application of long memory time series models. *Journal of Time Series Analysis* 4, 221–238.

Hausdorff, J.M. (2005). Gait variability: methods, modeling and meaning. *Journal of NeuroEngineering and Rehabilitation* 2, 19.

Hausdorff, J.M., Purdon, P.L., Peng, C.-K., Ladin, Z., Wei, J.Y., Goldberger, A.L. (1996). Fractal dynamics of human gait: stability of long-range correlations in stride interval fluctuations. *Journal of Applied Physiology* 80, 1448–1457.

Hof, A.L., Gazendam, M.G.J., Sinke, W.E. (2005). The condition for dynamic stability. *Journal of Biomechanics* 38, 1–8.

Holden, J.P., Cavanagh, P.R. (1991). The free moment of ground reaction in distance running and its changes with pronation. *Journal of Biomechanics* 24, 887–897.

Kadaba, M.P., Ramakrishnan, H.K., Wootten, M.E. (1990). Measurement of lower extremity kinematics during level walking. *Journal of Orthopaedic Research* 8, 383–392.

Kanko, R.M., Laende, E.K., Davis, E.M., Selbie, W.S., Deluzio, K.J. (2021). Concurrent assessment of gait kinematics using marker-based and markerless motion capture. *Journal of Biomechanics* 127, 110665.

Kinoshita, H. (1985). Effects of different loads and carrying systems on selected biomechanical parameters describing walking gait. *Ergonomics* 28, 1347–1362.

Knapik, J.J., Reynolds, K.L., Harman, E. (2004). Soldier load carriage: historical, physiological, biomechanical, and medical aspects. *Military Medicine* 169, 45–56.

Kobsar, D., Olson, C., Paranjape, R., Hadjistavropoulos, T., Barden, J.M. (2014). Evaluation of age-related differences in the stride-to-stride fluctuations, regularity and symmetry of gait using a waist-mounted tri-axial accelerometer. *Gait & Posture* 39, 553–557.

Li, Y., Wang, W., Crompton, R.H., Gunther, M.M. (2001). Free vertical moments and transverse forces in human walking and their role in relation to arm-swing. *Journal of Experimental Biology* 204, 47–58.

Lowry, K.A., Lokenvitz, N., Smiley-Oyen, A.L. (2012). Age- and speed-related differences in harmonic ratios during walking. *Gait & Posture* 35, 272–276.

Menz, H.B., Lord, S.R., Fitzpatrick, R.C. (2003). Acceleration patterns of the head and pelvis when walking on level and irregular surfaces. *Gait & Posture* 18, 35–46.

Moe-Nilssen, R., Helbostad, J.L. (2004). Estimation of gait cycle characteristics by trunk accelerometry. *Journal of Biomechanics* 37, 121–126.

Öberg, T., Karsznia, A., Öberg, K. (1993). Basic gait parameters: reference data for normal subjects, 10–79 years of age. *Journal of Rehabilitation Research and Development* 30, 210–223.

Owings, T.M., Grabiner, M.D. (2004). Step width variability, but not step length variability or step time variability, discriminates gait of healthy young and older adults during treadmill locomotion. *Journal of Biomechanics* 37, 935–938.

Pasciuto, I., Bergamini, E., Iosa, M., Vannozzi, G., Cappozzo, A. (2017). Overcoming the limitations of the Harmonic Ratio for the reliable assessment of gait symmetry. *Journal of Biomechanics* 53, 84–89.

Patton, A., Politis, D.N., White, H. (2009). Correction to "Automatic block-length selection for the dependent bootstrap". *Econometric Reviews* 28, 372–375.

Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E., Goldberger, A.L. (1994). Mosaic organization of DNA nucleotides. *Physical Review E* 49, 1685–1689.

Perry, J., Burnfield, J.M. (2010). *Gait Analysis: Normal and Pathological Function*, 2nd ed. SLACK.

Politis, D.N., White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews* 23, 53–70.

Qu, X., Yeo, J.C. (2011). Effects of load carriage and fatigue on gait characteristics. *Journal of Biomechanics* 44, 1259–1263.

Revi, D.A., Alvarez, A.M., Walsh, C.J., De Rossi, S.M.M., Awad, L.N. (2020). Indirect measurement of anterior-posterior ground reaction forces using a minimal set of wearable inertial sensors: from healthy to hemiparetic walking. *Journal of NeuroEngineering and Rehabilitation* 17, 82.

Richman, J.S., Moorman, J.R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology – Heart and Circulatory Physiology* 278, H2039–H2049.

Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D* 65, 117–134.

Schulz, B.W. (2017). A new measure of trip risk integrating minimum foot clearance and dynamic stability across the swing phase of gait. *Journal of Biomechanics* 55, 107–112.

Sekiya, N., Nagasaki, H. (1998). Reproducibility of the walking patterns of normal young adults: test-retest reliability of the walk ratio (step-length/step-rate). *Gait & Posture* 7, 225–227.

Terrier, P., Turner, V., Schutz, Y. (2005). GPS analysis of human locomotion: further evidence for long-range correlations in stride-to-stride fluctuations of gait parameters. *Human Movement Science* 24, 97–115.

Terrier, P., Dériaz, O. (2012). Persistent and anti-persistent pattern in stride-to-stride variability of treadmill walking: influence of rhythmic auditory cueing. *Human Movement Science* 31, 1585–1597.

van Schooten, K.S., Sloot, L.H., Bruijn, S.M., Kingma, H., Meijer, O.G., Pijnappels, M., van Dieën, J.H. (2011). Sensitivity of trunk variability and stability measures to balance impairments induced by galvanic vestibular stimulation during gait. *Gait & Posture* 33, 656–660.

Wang, Y., Srinivasan, M. (2014). Stepping in the direction of the fall: the next foot placement can be predicted from current upper body state in steady-state walking. *Biology Letters* 10, 20140405.

Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement*, 4th ed. Wiley.

Wu, S.-D., Wu, C.-W., Lin, S.-G., Lee, K.-Y., Peng, C.-K. (2014). Analysis of complex time series using refined composite multiscale entropy. *Physics Letters A* 378, 1369–1374.

Yentes, J.M., Raffalt, P.C. (2021). Entropy analysis in gait research: methodological considerations and recommendations. *Annals of Biomedical Engineering* 49, 979–990.

Zeni, J.A., Richards, J.G., Higginson, J.S. (2008). Two simple methods for determining gait events during treadmill and overground walking using kinematic data. *Gait & Posture* 27, 710–714.

Zifchock, R.A., Davis, I., Higginson, J., Royer, T. (2008). The symmetry angle: a novel, robust method of quantifying asymmetry. *Gait & Posture* 27, 622–627.

::: {custom-style="Note Box"}
**Note on the references** — the bibliographic details above were compiled without access to the publishers' websites. Check volume and page numbers against the journal before citing. Rosenblum et al. (2021) is cited in `REFERENCE_RANGES` (treadmill vs overground margins of stability) and its details should be added once checked.
:::
