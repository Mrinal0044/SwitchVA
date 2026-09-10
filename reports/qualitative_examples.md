# Qualitative Analysis: NSSG-DimNet vs Baseline (Oracle Ground Truth Conditioned)

Representative test case studies illustrating the comparative behavior of Switch-Point Gated Self-Attention (SP-GSA) and Neuro-Symbolic Graph Reasoning (H-NSG/RGAT):

### Example 1 (Sentence #347)
- **Sentence**: *wo bahut hi yogya or shalin insan rahe hai onke sanghars ka prtychh darshi raha hu wo apne chhote bhai k liye mai bjp k varisht netao se anurodh karunga ki onko onka haq de kr sammanit kiya jaye*
- **Target Aspect**: `netao`
- **Target Opinion**: `anurodh karunga ki onko onka haq de kr sammanit kiya jaye`
- **Ground Truth**: Valence = `0.850`, Arousal = `0.550`
- **Switch-Unaware Baseline**: Valence = `0.041` (Err: 0.809), Arousal = `0.770` (Err: 0.220)
- **NSSG-DimNet (Ours)**: Valence = `0.506` (Err: 0.344), Arousal = `0.577` (Err: 0.027)
- **Total Absolute Error Reduction**: `+0.657`

### Example 2 (Sentence #347)
- **Sentence**: *wo bahut hi yogya or shalin insan rahe hai onke sanghars ka prtychh darshi raha hu wo apne chhote bhai k liye mai bjp k varisht netao se anurodh karunga ki onko onka haq de kr sammanit kiya jaye*
- **Target Aspect**: `insan`
- **Target Opinion**: `bahut hi yogya or shalin`
- **Ground Truth**: Valence = `0.877`, Arousal = `0.573`
- **Switch-Unaware Baseline**: Valence = `0.133` (Err: 0.744), Arousal = `0.746` (Err: 0.173)
- **NSSG-DimNet (Ours)**: Valence = `0.618` (Err: 0.259), Arousal = `0.457` (Err: 0.117)
- **Total Absolute Error Reduction**: `+0.541`

### Example 3 (Sentence #227)
- **Sentence**: *kaash mein jitni parwaah kart hun logo ki aap bhi karte aapko toh sirf apne credit apne karma aur sirf aapne aap se matlab hota hai you are not able to treat right to one person who can give everything for the world overall mein permanent nahi but aap toh ho*
- **Target Aspect**: `aap`
- **Target Opinion**: `permanent nahi`
- **Ground Truth**: Valence = `0.200`, Arousal = `0.700`
- **Switch-Unaware Baseline**: Valence = `0.623` (Err: 0.423), Arousal = `0.539` (Err: 0.161)
- **NSSG-DimNet (Ours)**: Valence = `0.201` (Err: 0.001), Arousal = `0.750` (Err: 0.050)
- **Total Absolute Error Reduction**: `+0.533`

### Example 4 (Sentence #104)
- **Sentence**: *a n i sh pathan ds 4141 khan2 ye hum log isko toot bolte hai apne gaun men jab paj jati hai bahot mithi rahti hai bahot khate hai hum log*
- **Target Aspect**: `toot`
- **Target Opinion**: `bahot mithi rahti hai bahot khate hai`
- **Ground Truth**: Valence = `0.850`, Arousal = `0.500`
- **Switch-Unaware Baseline**: Valence = `0.276` (Err: 0.574), Arousal = `0.636` (Err: 0.137)
- **NSSG-DimNet (Ours)**: Valence = `0.687` (Err: 0.163), Arousal = `0.481` (Err: 0.019)
- **Total Absolute Error Reduction**: `+0.528`

### Example 5 (Sentence #226)
- **Sentence**: *desh hit mai apne abhi tak kya kiya sirf nafrat key ilawa apka logo ka ek he agenda sirf ek jatti dhram ko gali 2 logo mai nafrat bato yeah hai apki deshbhaqti*
- **Target Aspect**: `deshbhaqti`
- **Target Opinion**: `apki deshbhaqti`
- **Ground Truth**: Valence = `0.120`, Arousal = `0.780`
- **Switch-Unaware Baseline**: Valence = `0.420` (Err: 0.300), Arousal = `0.451` (Err: 0.329)
- **NSSG-DimNet (Ours)**: Valence = `0.071` (Err: 0.049), Arousal = `0.857` (Err: 0.077)
- **Total Absolute Error Reduction**: `+0.503`

### Example 6 (Sentence #236)
- **Sentence**: *izzhat se kamana or izzhat se halal kamana or apne bachon ko kahlana b to abadt hay sir*
- **Target Aspect**: `halal kamana`
- **Target Opinion**: `izzhat se halal kamana`
- **Ground Truth**: Valence = `0.950`, Arousal = `0.550`
- **Switch-Unaware Baseline**: Valence = `0.111` (Err: 0.839), Arousal = `0.763` (Err: 0.213)
- **NSSG-DimNet (Ours)**: Valence = `0.414` (Err: 0.536), Arousal = `0.630` (Err: 0.081)
- **Total Absolute Error Reduction**: `+0.435`

