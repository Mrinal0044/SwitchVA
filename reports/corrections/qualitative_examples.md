# Qualitative Analysis: NSSG-DimNet vs Baseline (Oracle Ground Truth Conditioned)

Representative test case studies illustrating the comparative behavior of Switch-Point Gated Self-Attention (SP-GSA) and Neuro-Symbolic Graph Reasoning (H-NSG/RGAT):

### Example 1 (Sentence #326)
- **Sentence**: *koye apne pyare hathon sey chai shai hii pilla do*
- **Target Aspect**: `chai`
- **Target Opinion**: `pyare hathon sey chai shai hii pilla do`
- **Ground Truth**: Valence = `0.693`, Arousal = `0.481`
- **Switch-Unaware Baseline**: Valence = `0.087` (Err: 0.606), Arousal = `0.746` (Err: 0.265)
- **NSSG-DimNet (Ours)**: Valence = `0.397` (Err: 0.296), Arousal = `0.522` (Err: 0.041)
- **Total Absolute Error Reduction**: `+0.534`

### Example 2 (Sentence #347)
- **Sentence**: *wo bahut hi yogya or shalin insan rahe hai onke sanghars ka prtychh darshi raha hu wo apne chhote bhai k liye mai bjp k varisht netao se anurodh karunga ki onko onka haq de kr sammanit kiya jaye*
- **Target Aspect**: `insan`
- **Target Opinion**: `bahut hi yogya or shalin`
- **Ground Truth**: Valence = `0.877`, Arousal = `0.573`
- **Switch-Unaware Baseline**: Valence = `0.162` (Err: 0.715), Arousal = `0.765` (Err: 0.192)
- **NSSG-DimNet (Ours)**: Valence = `0.494` (Err: 0.383), Arousal = `0.546` (Err: 0.027)
- **Total Absolute Error Reduction**: `+0.497`

### Example 3 (Sentence #286)
- **Sentence**: *aruem nothing international armys ed sheeran ko trend krti hain aur philippines armys apne singer ko trend kr rahi hain isliye hum bhi krte hain*
- **Target Aspect**: `ed sheeran`
- **Target Opinion**: `trend krti hain`
- **Ground Truth**: Valence = `0.562`, Arousal = `0.429`
- **Switch-Unaware Baseline**: Valence = `0.179` (Err: 0.383), Arousal = `0.772` (Err: 0.343)
- **NSSG-DimNet (Ours)**: Valence = `0.442` (Err: 0.120), Arousal = `0.561` (Err: 0.133)
- **Total Absolute Error Reduction**: `+0.473`

### Example 4 (Sentence #128)
- **Sentence**: *farzi kisan andolan ke liye koi permission nhi chahiye democracy ha bhai shaheen bagh me farzi andolan or roads blok kr skte ha par hindu apne desh me equal laws ki maang kre to secularism khatre me agya isupportashwiniupadhyay isupportashwiniupadhyay*
- **Target Aspect**: `andolan`
- **Target Opinion**: `farzi kisan andolan ke liye koi permission nhi chahiye`
- **Ground Truth**: Valence = `0.150`, Arousal = `0.800`
- **Switch-Unaware Baseline**: Valence = `0.581` (Err: 0.431), Arousal = `0.436` (Err: 0.363)
- **NSSG-DimNet (Ours)**: Valence = `0.352` (Err: 0.202), Arousal = `0.677` (Err: 0.123)
- **Total Absolute Error Reduction**: `+0.469`

### Example 5 (Sentence #408)
- **Sentence**: *i a meri sis meri dost mn apne hr dost ke st hr waqt kharra ho*
- **Target Aspect**: `dost`
- **Target Opinion**: `apne hr dost ke st hr waqt kharra ho`
- **Ground Truth**: Valence = `0.842`, Arousal = `0.564`
- **Switch-Unaware Baseline**: Valence = `0.073` (Err: 0.769), Arousal = `0.737` (Err: 0.172)
- **NSSG-DimNet (Ours)**: Valence = `0.417` (Err: 0.425), Arousal = `0.616` (Err: 0.052)
- **Total Absolute Error Reduction**: `+0.465`

### Example 6 (Sentence #594)
- **Sentence**: *mla johar dada apse request hai is latter ki tarah aap bhi apne latter ped pe likh ke cm sab se avgat karane ki kirpa kare*
- **Target Aspect**: `cm`
- **Target Opinion**: `avgat karane ki kirpa kare`
- **Ground Truth**: Valence = `0.600`, Arousal = `0.350`
- **Switch-Unaware Baseline**: Valence = `0.229` (Err: 0.371), Arousal = `0.711` (Err: 0.360)
- **NSSG-DimNet (Ours)**: Valence = `0.421` (Err: 0.179), Arousal = `0.533` (Err: 0.183)
- **Total Absolute Error Reduction**: `+0.370`

