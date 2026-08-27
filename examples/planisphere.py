"""
The whole sky at once, drawn as the instrument that shows it: a planisphere,
the visible hemisphere pressed into a disc with the horizon for a rim and the
zenith at the centre.

The projection is stereographic from the point underfoot, the classic choice
for star charts and not only for its looks: it is conformal, so a
constellation near the rim is enlarged but not sheared, and the figures stay
recognisable everywhere on the disc. The compass points sit on the rim -- with
east to the *left* of north, because this is the view lying on your back
looking up, not the view of a country looked down upon. Stars wheel about the
celestial pole, which from Oxford stands a little over halfway from rim to
centre, marked by Polaris.

A planisphere is set to a time by turning a wheel; this one takes `--when` (or
the clock) and turns itself. The hour picks the local sidereal time, the
sidereal time turns the sky, and left to play, the chart wheels through one
whole *sidereal* day -- 23 hours and 56 minutes, the time the stars take to
come back round, four minutes short of the time the sun does -- so the loop is
seamless. The sun is computed alongside the stars and the sky answers it:
civil, nautical and astronomical twilight each wash out the stars to a deeper
magnitude, dawn drowns all but the brightest, and dusk hands them back.

The stars are the Yale Bright Star Catalogue by way of the d3-celestial
project (BSD-3, Olaf Frohn), packed below to a tenth of a degree: every star
to magnitude 4.5 plus every star a constellation figure touches, 1041 in all,
each with its magnitude and its B-V colour index, so Rigel comes out
blue-white and Betelgeuse orange. The figures are d3-celestial's IAU
constellation lines, stored as indices into the same catalogue. One star is an
interloper: Mira, the pulsating heart of Cetus, spends most of its cycle too
faint for the catalogue's cut but sits on the figure of the whale, so it is
drawn -- as star atlases have always drawn it -- at a middling brightness.
The catalogue's positions are epoch J2000; a quarter century of precession has
since moved the sky a third of a degree, a fraction of one braille dot here.

There are no planets. There never are on a planisphere: they wander along the
ecliptic at their own several paces, so no chart fixed to the stars can hold
them. The moon is a wanderer too. What this disc shows is exactly what the
instrument can promise: the fixed stars, the figures, and the sun that decides
whether you can see them.

By Claude Fable 5.
"""

import datetime

import tyro
import numpy as np

import matthewplotlib as mp


# Where the sky is stood under, unless the command line says otherwise.
OXFORD_LATITUDE = 51.7520       # degrees north
OXFORD_LONGITUDE = -1.2577      # degrees east

# The disc has radius one and the frame reaches a little past it, leaving room
# for the compass letters to sit on the rim.
REACH = 1.12

# The sky's colour is a single reading of this ramp per frame, keyed on how
# far the sun stands above or below the horizon: night, the three twilights,
# and day. The stops are the conventional twilight boundaries.
SKY_SUN_ALTITUDE = np.array([-24.0, -18.0, -12.0, -6.0, -3.0, 0.0, 8.0])
SKY_COLOR = np.array([
    (8, 9, 20),         # night
    (10, 12, 28),       # end of astronomical twilight
    (19, 26, 56),       # end of nautical twilight
    (46, 62, 108),      # end of civil twilight
    (84, 110, 158),     # sun close under the horizon
    (126, 156, 202),    # sunrise
    (148, 188, 232),    # day
])

# How deep into the magnitudes the eye can go, for the same sun altitudes:
# magnitude 6.6 is the darkest-sky limit, and by the time the sun is up only
# the sun is left. Between the stops the limit slides, so stars do not vanish
# as a block but drown one magnitude at a time.
LIMIT_SUN_ALTITUDE = np.array([-18.0, -12.0, -8.0, -4.0, 0.0, 4.0])
LIMIT_MAGNITUDE = np.array([6.6, 5.6, 4.0, 1.5, -1.0, -4.0])

# A star's tint from its B-V colour index: hot blue-white through white and
# yellow to the deep orange-red of the coolest giants.
TINT_BV = np.array([-0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.4])
TINT_COLOR = np.array([
    (155, 176, 255),
    (202, 215, 255),
    (248, 247, 255),
    (255, 244, 214),
    (255, 210, 161),
    (255, 178, 110),
    (255, 138, 78),
])

SPACE = np.array([6, 7, 16])            # past the rim, no sky at all
HORIZON_RING = np.array([64, 72, 92])   # the rim itself
GLOW = np.array([255, 158, 88])         # sunrise and sunset on the rim
FIGURE = np.array([62, 80, 116])        # constellation lines

# The figures and their labels are drawn only while the sky is dark enough to
# show the stars they join: they fade as the limiting magnitude falls towards
# this one, and are gone when it passes.
FIGURE_MAGNITUDE = 4.4
SUN_DISC = np.array([255, 243, 205])
COMPASS = (176, 184, 200)
LABEL = (110, 122, 146)

# Sunrise and sunset are conventionally the moment the sun's upper limb
# touches the horizon, which refraction and the sun's own radius put at this
# altitude of its centre.
SUNRISE_ALTITUDE = -0.833

# The catalogue: 1041 stars, brightest first, ten hex digits each --
# [0:3] right ascension in tenths of a degree, [3:6] declination in
# tenths of a degree from -90, [6:8] magnitude in twentieths of a
# magnitude from -1.5, [8:10] B-V colour index in twentieths from -0.5.
# Every star to magnitude 4.5, plus every star the figures below touch,
# plus Mira.
STARS = """\
3f52dd010a3c0175120d85b4441d238971241e18ae85081f0a318550201a3123322209
47c3b826130f414827073783ce272883e1282a05ba13dd2d0e74a10d2d052b24292f29
7dd31432059aa27c332f48b49c351ed7425c350d77f12f3705c20549370c897124391c
5f13fc39084172623c064704c33e0b7561493e2aa4a2113e0532d3c33f063304a23f07
5670cb3f0b3493784006cf91ae410935437141064c81ab410778f5b4410a1ff5774214
ac822c420967b5ee421f42f27c431781557143084e81314322a531d64312383545440c
9da0d244273e2428450a520161450bbf814d450817c70145173bd2d0460558b32d4627
13e46f462160e44a462106d2d0471eb1627d4707845218471e0154a747090ae4e8472a
36532347078b366a4727d4f1af472aa4d402480d1d651e480a13652b48256ed416490c
08e5e3490977019a4a0a4b91f44a055711334a0e92148f4a0b55a1d24b2b7da5a94b0b
bf05174b170655b94b21a845874b2833e3814b060175d34c1280216d4c079612a24c08
9dd22d4c2189d1aa4c0788d1de4d076775b84d0b8a44934d1dcbc3e74e28a601fe4e07
0421dd4e206f959d4e0ba102e74f0bd8349d4f2b45625f4f08c7c5f64f0f57d15e4f07
c2c4d8501ed8641c500a1c83ad512b9bd31a510b8291ab5106695451510d96e2be5109
3402d2520e71d18952077302d55208b29259520b8f532652099393c4532111f454530d
35122f53083834f8530875e29a531c0d75de530d82743c54168c61d554062e64d05428
651196541c7650d15406a4320f5406445211542aac025a5426b963ee542898435f552a
99c5eb551c6471005506770376551134734955067d1215550b8b32e4550da633b25521
302351560d9ac45b561da1b414562172e1395606a4258f561d9221e856063352b4561a
9c74c0561704007f56169ba26a5606ad2286561e02141c57064c3291571395410a5710
2494c3570fa351595728a46191570723947557087a33f2571dcc42e3570e12911c5710
b92547570a3bd465572a8f90d5570ad1212957261be1f1580db3a2b2581245e3d75808
794503580895d27f580625351458069952845810c9d34c581b1ce59b58185bc0f9580f
d504b2591b40118a59227532df590acf2381591d2532fd592a34c45759075b9472591a
a9b2545a1e7cd29c5a1ca6d1f35a14b3040f5a0a1444e25a0d68a5415a218fe6525a0b
9e22085a06cd520e5a0822d5625a083b72575a074222965a087212a25a252f353a5b15
77c0db5b068845035b0eb6f49c5b20bed2f05b1a3f247f5b266145235b2ab416295b1e
ab82145c2a53a3bf5c1e6582e25c236cb10e5c09c161ab5c1e36d21e5c215445645c0e
9f31545c29a1c47c5c0c8c81df5d0657b4dc5d2959414a5d29a1c4f45d273e21d45d08
5985895d14a0c6155d08afe2765d082fe5205e0789e0fa5e0f2d53ca5e142fc2a45e27
9f03e25e21a732125e22c6e4b25e1eddc68c5e1f8ff1ee5e0598e3555f1dac13675f1d
c966465f063fc1195f0ebd437c5f094631d35f288442795f208d42875f2bb1f4cb5f09
23809e5f2a0624b95f23a2d28a5f06d6d2e65f0b30e2e260085fe0c860099085d26021
2ad15e600862c11b60083a9465602aa3515060070a51b1601c1cf508602972f5be600c
a141d46013a8a322601eb3326f602127c113611c69641e610a49528b612211e6016107
32b36c61053f540561134fc5e3611bb623a361109031c561065253c462187f537e620c
3463e7620760711f62297933a66229cff5ca62292a0423620e0dd1d3622925a4016208
81a1e362068e917b621cc295ee621cd4c3f0620811b4ac62149602046206a6a4996219
c280ee620d561136630660646e6310b323536308607531630b07b5c663160ab31e6321
4a817263068f14d1631d1983a4630c81a1db63075ee42c64099cb509641c1042e56419
41e26d642d6a04cf64268cf518641dacf1b86406d5d183640c40123f640844c4606411
d6061a641fbb54476429d6147a641d22e322641c5ad3e764145d41626409b094d2640a
b1c2b16421cfd3c2640c2a0444651e4d93e065286c5245651d9463626509a542ea650f
28523265083632f0650c8651b76506ac165b6514bce0ee651903132c65221551816508
6a22f065209e32086506489478651d54f55c650a8844b465249012196529447429660c
be5307661c0f556a662431a340660835d2a466146f1396661473d12866260d2332661f
4084d8660c51517366075931ef661192726b6625a4412566081a949566082003de661c
5f6308661e7a40b96622a680fd66210e541d661d23d475660948b208662c9e81dc6626
d7f52b66086e40e9670dc164166712289420671e5955fb671193d41e670ba9d18f6708
d2c38467129104a7671092b25a670683f6086709c4113b67232d83bc680751d2386806
b9843d6824d922b0682205c59f6806122180681b2f451f68215b7113681e6e55626822
cb22dd68101f32aa682aa864a8681dda53a5681c2e039c680637b2f6681149d1ee681e
9493b1680da9e3e4680dbac3c4681b214325681c23247568083835a3681ec5a53b682a
2063e569088a8397690aa7c5bd6922cb607e691ed68338692b11731d69219974446910
c7350069125501ad6921a6e39f690b342113691736e2b3691e3cc33e6908b2e2ab691e
b6c589690dd3257b690b1ab5b3692c23352e6912291433691e34f36a69064f00ef6921
5161b269179dc1366929c1b4236909c912a4691ecf648169134300c36a1e45a49a6a1e
5c55d26a106621376a1d89b40d6a0bc2f3256a0a1da5456a1e6614da6a1f69f4bf6a16
1e02626a1547b2786a079213ed6a0fb4d59a6a1dbda5576a232b12526a1d6221396a10
93548b6a0add85556a1e1313a06a0a5754f46a0b6c16396a2a9ad3986a0aa595506a06
61d2dc6b278301df6b068ac06e6b272300fc6b212314c76b0a29f4246b1d53a1266b08
62e3e16b076391a26b107590b36b07a9f4a46b0aaae2b16b0eb9a6436b1c27b1dd6b20
3641856b0d3bb2366b1b6011df6b0b75c63e6b0876619f6b0b9574216b14ac745e6b21
ae43326b2408e5056b0d2b72f56b2033c2216b219871076b209b406f6b1ca834f96b25
d1a3766b092354786b095231b86b0a8311c46b068a034b6b129582606b06ba538e6b17
0181bb6c1e11c4456c095ca4886c228e819d6c09d941c06c1e1b932b6c2040b2926c2d
56a39b6c0973a37d6c0bbaf4e36c1e5aa3796c246a71636c077ec1fa6c222603c06c0b
4f035d6c0a74e18e6c068d71ad6c079232f06c1e9915536c079f74b96c0ab582d26c0f
c763b86c150421cf6d0e0e41996d1d1b45946d192b33626d064830ae6d1f6871366d22
9712b56d09a8e3a16d0b0ab15c6d084813246d1e4882626d0d52043a6d20c475206d0b
1356586d0a3e02c36d1f26e5616d0a3821d86d215485266d135621156d06b5b1c76d08
b5e1ee6d08bdf5616d27db32bb6d202942306d2736f50b6d214440dc6d195142236d1d
7251786d079492346d09bb90ab6d09d2d1d16d1ed584706d1f2554ea6e0aca354c6e1c
12c2b16e293a93456e246190a06e117dd5aa6e0dda613e6e1254c0ec6e0d6ae3ed6e12
97e2c16e0c10357f6e082ca3636e074244526e1c43f2786e0789d20a6e079655ce6e15
99218e6e20af80ba6e21c014b46e1252e26f6e2371d28d6e11b2141b6e20b2634b6e20
1574d66f0a2f75e06f1c5254a46f1e7901486f06a9a39d6f1bc0b3f56f08e0e3c96f12
2763406f116e53c56f2887358b6f14d3c3836f081d95746f164de0836f1251f1da6f1b
6a73c06f0981a4226f288672096f098931966f07d602fc6f292e540b6f214744916f29
6b02d36f0e7340dc6f2a7361046f078a12246f259f51706f2785c3486f148e01bf6f07
8f61386f0cd2d5cc6f1a07647770201620d5700b18f38770061c8298700d40b30c7026
59c134700a6712cd7020b1653b7026c5d2d8700ac8544a702034a3e1701d944439702a
0f2522701519a57070144951b47007ad4199701eb3b1fb70211921f5701e4232e87008
50e1d6700c5c72f0701c635072702a6e6120701c9260ed7021b3a209700bd7c174701d
18f0d9700927d568701d38a3e4700d7153db701dbac1e1701fc3a277702bd2e1cf7029
32a336711d3cc44e71089512dd711ec2b2877113d57056710edde3bc711420f4057120
236473710950e3bd710a7190fe711191c4be710795a4917123cbe4847112d0c4fd7127
43837f710aa314f7710addf53f710938e46d711b4634c2711096b3127113a382927110
0525f9710d23e21a711dd0e336711e85431d722485d551720c9bb2237229d0a5be7210
d4a276720881a22b7228afe451721433b3bf72074a829f72186224f3721c9b552c720a
d593fe7214d7023f721d20b5db7212297463720d9d36b8721cc025fa720ec900f67214
1aa503721123729c721349f1a3720797021472069a42cb720eac064d7208b023557220
b0f1167207b144f5722ac2a4b7721fc7a50e720cd9e34872290320fb73164864a57320
7bc49b731581023a7312838393730c97654573099cc07d731fb322117314cbb5d07337
cc357173080ac6e3732216f1a7730719a2f9730873606b730875c52273169ae2297307
a5e303730cda232973200bd0d3731426857c730a2b13ea730e2b4521732130532c7306
4e9534732978e1f2730e8796797327cc05e773130ae55c730a1073e0731d1f41d57318
2162ac730828018173102df61b730a5423fb730da512027320a9045c7312ced60a7312
09d3d3731d19c3e9731027f3dd73092b7401730c2c246a73087af19173068f81a57308
b5c1c47311c2d425731fdc03c4731f29a468740f5a36b1742862024d7427a3a2597412
bd95ba740cc862dc741ccfd4d07413d8b1d1741221e388741631f300740563e158741e
6f6231740893868e740b9ae2de741ccf42f97408d33241740add9535740809325e7407
1723d97409233479740823b20c7409298437740b38454f742c51c3a6740666747b740a
6ce37c741e6ee10674077582e27412867150740cbe4307741d51612e740887703f7424
9074fa74109732b3741bc18379741d21d56674092bd2bf742a33f43e74333aa4ab741e
5242fd741c59546a742971010b74107b7433741382023a74078b51d07407a2a304740b
4603dd752678d19b75258751be7513a1c27a751baa1107750fabe4ed752105c4d57508
0604a9751b0b25ab750d35e0f3750e3d429a75054e407d752156f145752a5a2196750d
8381bc75169291da7526a383ad7528afc4fc750eb0243a750dcbd4327521d2e561752c
1df449751f2d73dd750a3db172750a4135cc751b4c40d675085293be750974b49f7521
8f72577520a476e6750aac611d7527b4b5017523cbe23a750904f10e750906c145750a
14c3dc751c20e5647525264461751f2dc34d750f30f303750837e22375074102d97509
4c5366751d8431e87506b7e3777508005348752b3ad225751e44928a75076721de750c
aa24547507bb722375077b734d760abce68d7609dbb2b67627dcc20a760811c1b5762a
23e613762f3453e3760737644f76163bf3b2760e4b639b76235f03e876278d1399761f
9e73ea7608a2b2b17612b86438761ab89433761fc8016e760e08f46e761d4922817609
4c72c47607b1a0e47615cdf15e761040116c761c4364b2762386e1f97606a434897627
a884b27612c7b4e17608d8e676761ada73287607da923f762039741876073e32ce7621
43c1c6762543e27c76074d31f1762167844e760b8b226c762593e3ce7616a90391760b
d1f58e761edb946e761623530b772a27e337771a2e759e770ab4450b7707bbf26f772b
bf44c67725c2541b7710c2f352772b0252c7772b07a3d0772828113377202bd1e17711
3b55d2770b46f2a577144cc1f8772a4de215770e88618b7706b6a47b77280fe3bb7725
1a7240771e30d30d77083a241277065113a577225461e77717635144772a64e05f7706
69c35f770e9b02ad770db496627723b773ce7722175626770d24a366771730f3a17721
55c588771068f2a0770a71c18977079a41a87709d6723b770919d2ca77141f54a67729
2ea39577253ca23e77073d63cd770a4ac19877065530ae771658448a772259d58d770b
6f21c0772488b4ad77118c33597710c723e87715cf31f97725070567780925a1167828
29a420780f3fc39c78205f03807809904133780d99c2bc781ed843aa7808d8b297781c
131649780d2ac25a781d3f240878214391b0781049e1ff78065ef4e4780e89a428780a
b7d57a7812c6028a782acbc4a37814de3396780ede52f378090000f478083503ad7808
36b152782052514c780761015478086ea44e78158144337814bdf49a7823c66312781d
c962aa781ccf923a780bd075117826d4953f782415c36678260b34b1782058f21c7826
932449780bc2617d7810d280fa7809aa81b8781eb582e4780cd30533780807d51f7907
85658a790f59c4f0791c2f9221792241a68679255c85a1790bd195557908d21573790c
25511e792a21f1f1791e9a029a790fb371ef791fba8475790924064d7a0b6921297a15
9464897a1aafb5107a0eb7e28b7a08df426b7a0a66a2117a1e94e52d7a1511c3a47a1d
1de5107a2045d5707a0a6821147a1eae01dd7a1eb193ae7a0d2e13ea7b0c38e44d7b0f
2f941e7b0937510d7b1e4722687b089681987b1c0b34567b1e3ea3e77b0551c45b7b0a
92d22c7b1dbbb49a7b0e0b847a7b1f3932ef7b0bad52f27b0cc512417b1c11f0e07c1d
13725f7c071bf5117c0b9930717c2c6ce3227c09af63297c11bae27d7c19c683e97c0f
1c13dd7c082fc1457c156e22cc7c1dc7b2427c0b61d4d67c0f9701c07d0fc3c32a7d10
0c74957d0bb2a1df7d093b36397d0ab4b45a7d09a943327d12c002d27d12d0b1e77e1a
c821ec7e0bd233927e073064207e106b03177e29de22d27e1ab1f2117f120e13c17f25
a475ac7f10b132a17f26b492877f151863bc7f1c7070768009b4e2c6801ec35232801e
98c67a8112dbd391810bdee3a7813cded55481201bf103820d9644af8209b6128f820f
cc524f820b2c12118312dad3ba83222a51c283065c9333830b6c1132831f39e0988418
62837e84076ac2c88413687119840ec311cc84111cb12f8411c2d42585146fe2d8850a
33e089862162636986090b83d08610192163861218a17788102f50bb881e1961888a15
64d05f8b1d2e20978b28b081d28e087945038e111c151e9427
"""

# The IAU constellation figures: per line, a three-letter tag and
# comma-separated polylines of dash-joined star indices, hex, into the
# catalogue above. A long constellation continues on further lines with
# its tag repeated, and a polyline split across lines rejoins on its
# repeated index.
FIGURES = """\
And 03d-037-0e1-036,32a-232-2ed-0e1-2ec-2c8-150,2c8-263-19e
And 037-1ba-3a2-2a4-13c,263-3f3
Ant 39b-2ba-3b5
Aps 1a6-3cf-28f-1be
Aql 075-00b-16e-0db-1c6-0f8-0ab-00b-0f8-10d
Aqr 192-3da-09e-0a4-1c0-157-21e-177-336-15f,09e-2c6,0a4-26a,157-3e3
Aqr 1f3-336-3e6
Ara 0ea-146-189-0c4-22d-08e-08d
Ari 147-031-068-1c8
Aur 029-005-0ce-06f-01b-06a-029-171-005-0b5-162
Boo 392-06d-002-138-0b7-11a-112-051-002-191,0b7-26c-3a3-21d-26c
CMa 02f-000-0b3-025-118-016-0b2,058-025,000-30f-244-236-30f
CMi 007-099
CVn 40f-297
Cae 3fa-34e-3f8-3a5
Cam 345-215-2a9-3af-31e-278,2a9-3dd-3a6
Cap 2d5-0b9-3e0-256-251-18b-091-165-2bd-23a-2d5
Car 0ca-001-01c-0e5-078-0e8-0fd-042-027-111-020-02c-042
Car 078-3ba-400-3b0-1e2-190-0e8
Cas 0f4-06c-03f-047-04a
Cen 1d2-063-1d6-040-04b-05f-114-103-035-04f-0c6,103-07b,014-04b-00a
Cen 1d6-1f8-3fc
Cep 27b-105-059-291-26e-0ff-231-11f-0d5-0d9-059,0d9-11f
Cet 113-3ec-2ca-2af-3d4-05d-113-234-399-178-117-033-132-110-141-178
Cha 220-247-40c-296-3ed-247
Cir 230-0cf-37c
Cnc 2aa-1e8-3c5-216,1e8-129
Col 1b3-0c2-069-1bc,0c2-1ed
Com 2e2-28b-301
CrA 3e7-290-24a-242-3ad-3dc-40e-3bb
CrB 25d-158-043-19d-3b1-25e-3f5
Crt 3d0-3e5-134-238-365-3ff-229-3d6-404,134-229
Cru 012-081,00c-018
Crv 211-0b4-064-0a3-06b-0b4
Cyg 0d4-05b-046-093-181-199,013-046-1d0-0b8
Del 219-152-18a-403-349-152
Dor 2a8-0e7-17d-2ef-3c0-17d-3d5-0e7
Dra 175-048-082-3e9-175-0bc-282-0cc-077-20c-0e6-15a-1b5-1a1,282-130,0bc-1b0
Equ 1dc-375-3d3
Eri 07e-208-1e0-21b-0a5-343-122-16f-1cc-295-369-235-166-2a7-27e-19b-1f4-12d
Eri 12d-269-3ab-2a6-097-243-294-133-161-008
For 196-356-3cd
Gem 0e9-094-0ba-330-017-010-228-11e-209-02b-0f6,11e-13a
Gru 24b-11c-03a-252-01e-03a,1fb-3e1-376-0b0
Her 179-07f-085-116-275-1d9-28e-3b6,085-1da,116-0c9
Her 1bf-262-0c9-1da-0c5-109-167-1ae,080-07f
Hor 1b1-40b-409-408-402-3f4
Hya 0fa-300-2cf-359-25b-0fa-0be-1ce-1d1-030-246-149-1a4-0bf-12a-2c3-0a9-0dd
Hya 0dd-33e
Hyi 086-0e0-24c-233-3cc-092
Ind 0c1-39d-15b-32e-329-0c1
LMi 385-3d8-194-274-385-3a4
Lac 341-182-2fa-3a8-3a1-398-2fa-3a9-341,3a1-397-260
Leo 015-115-032-060-03e-0f2-015,032-10c-1c9-0a7
Lep 3c9-16a-12e-062-0e4-0d1-084-13e-17e,30d-0e4-2c1
Lib 0de-07c-066-1d8-145-159,07c-1d8
Lup 1f9-3c6-139-0d6-06e-04e-104-2b4-0f9-083-108-280,0d6-083
Lyn 34f-2fe-3b9-2a0-1ee-1a0-0c7
Lyr 2f7-3b2-004-2f7-285-0df-125-2f7
Men 3fd-405-40d-40a
Mic 3ef-401-3e2-3d7-3cb-3ef
Mon 1e6-310-261-17f-200,261-37a-321-36d-3c4
Mus 151-22a-071-0b6-14a-1ad-071
Nor 3d9-367-20d-3c2-3d9
Oct 2d7-257-176-2d7
Oph 0ef-17c-07d-03b-0d2-1a2-076-0d7-05e-056,0d2-05e-2c5-281-37d-3ac
Oph 07d-056-0e2-2bb
Ori 338-320-3be-358-24e-009-01a-3bd,36b-169-15c-0d0-2fc-3bd-227-3bf-3e4
Ori 006-0f5-049-01a-0fc-009-01f-038,01f-01d-049
Pav 02d-10a-131-284-304-2ea-14b-20e-1fa-10a-27c
Peg 2be-0a1-057-036-089-05c-276-106-127-052,05c-057-121-1fc-18c-25f
Per 1a8-08c-1fd-09c-184-0b1-2db-022-09f-183-1df-21f-193-03c-3b8-0ec-3ce-410
Per 410-03c,29c-24d-1ec-0b1,21f-23f-207
Phe 054-0eb-101-1de-1e5-0eb-1c7-054
Pic 0da-38e-1b2
PsA 26f-011-277-368-2c7-396-306-3f7-396-26f
Psc 3c8-39a-3db-3c8-3c3-14c-2a5-19f-3b7-355-3e8-407-2ae-34c-21a-258-2b7-3f9
Psc 3f9-168-3f1-38a-3f2-258,168-37e
Pup 0ca-073-3c1-197-0f3-273-08a-041-020,0f3-32b-1e7-3c1
Pyx 041-1f7-15d-210
Ret 0f1-34d-3aa-1a7-0f1
Scl 2c9-3b4-337-31c
Sco 09b-04c-061,04c-09d-00f-087-04d-0af-14f-0ee-028-0aa-053-019
Sct 1b9-283-3d1-3ca-1b9
Ser 155-39c-23d-1b7-155-198-067-16c-076,056-12c-0ef-3df-0d8-3bc
Sex 37b-3fb-406-3fe
Sge 327-15e-120,328-15e
Sgr 0bd-023-074-088-1af,1f0-1f1-065-0cd-088
Sgr 250-318-3d2-3b3-3f6-3eb-034-0cd-074-0a8-023-065-0f0-034-180-098-3ee-1db
Sgr 1db-3a0,180-126-3ea-034
Tau 0a6-00d-100-153-185-128-01b,153-102-173-1d4,173-148-2c0
Tel 39f-11b-241
TrA 02a-08b-095-02a
Tri 107-0ac-214-107
Tuc 096-203-307-289-38c-39e-096
UMa 0ed-024-050-055-0ed-021-045-026,055-164-119-195,164-0ad-0bb,0ad-10e
UMa 024-154-0f7-18f-050,050-3a7-0cb-0c3,137-0cb
UMi 2c4-3f0-0ae-039-2c4-27a-303-02e
Vel 02c-05a-124-070-1b4-144-044-020
Vir 21c-13f-1cf-079-319-00e-22e-1c4,090-0fe-079,319-0fb-28d-174
Vol 204-187-2ff-1f6-18d-2ff-204
Vul 3de-354-3ae-3c7-393
"""


def main(
    when: str | None = None,
    hours: float = 23.9345,
    num_frames: int = 96,
    fps: float = 10.0,
    width: int = 64,
    latitude: float = OXFORD_LATITUDE,
    longitude: float = OXFORD_LONGITUDE,
    place: str = "Oxford",
    figures: bool = True,
    labels: bool = False,
    loop: bool = True,
    save: str | None = None,
):
    """The sky overhead, as a planisphere that turns with the clock.

    By default the chart starts now, over Oxford, and wheels through one
    sidereal day, so it loops seamlessly. Pass `--when 2026-12-25T22:00` for
    another hour of another night, `--labels` for the constellation names, and
    `--num-frames 1` for a still chart of the moment itself.
    """
    if when is None:
        start = datetime.datetime.now()
    else:
        start = datetime.datetime.fromisoformat(when)
    if start.tzinfo is None:
        start = start.astimezone()

    step = datetime.timedelta(hours=hours / num_frames)
    frames = mp.tstack(*[
        chart(
            when=start + i * step,
            width=width,
            latitude=latitude,
            longitude=longitude,
            place=place,
            figures=figures,
            labels=labels,
        )
        for i in range(num_frames)
    ], fps=fps)
    if num_frames == 1:
        print(frames[0])
    else:
        frames.play(loop=loop)

    if save:
        if save.endswith(".png"):
            frames[0].saveimg(save, bgcolor=tuple(SPACE))
        else:
            frames.savegif(save, bgcolor=tuple(SPACE))


def chart(
    when: datetime.datetime,
    width: int,
    latitude: float,
    longitude: float,
    place: str,
    figures: bool,
    labels: bool,
) -> mp.plot:
    """One reading of the sky: the disc, its letters, and the caption."""
    height = width // 2
    jd = julian_date(when)
    ra, dec, mag, bv = STAR_TABLE
    alt, az = altaz(
        ra_deg=ra, dec_deg=dec, jd=jd, lat_deg=latitude, lon_deg=longitude,
    )
    sun_ra, sun_dec = sun_radec(jd)
    sun_alt, sun_az = altaz(
        ra_deg=np.array([sun_ra]), dec_deg=np.array([sun_dec]),
        jd=jd, lat_deg=latitude, lon_deg=longitude,
    )
    sun_alt, sun_az = float(sun_alt[0]), float(sun_az[0])
    limit = float(np.interp(sun_alt, LIMIT_SUN_ALTITUDE, LIMIT_MAGNITUDE))
    sky = ramp(SKY_SUN_ALTITUDE, SKY_COLOR, np.array(sun_alt))

    layers = [
        backdrop(width=width, sun_alt=sun_alt, sun_az=sun_az, sky=sky),
        horizon(width),
    ]
    if figures:
        stroked = figure_series(alt=alt, az=az, limit=limit, sky=sky)
        if stroked:
            layers.append(disc_strokes(stroked, width=width))
    starred = star_series(
        alt=alt, az=az, mag=mag, bv=bv, limit=limit, sky=sky, width=width,
    )
    if starred:
        layers.append(disc_points(starred, width=width))
    if sun_alt > SUNRISE_ALTITUDE:
        layers.append(disc_points(
            sun_series(sun_alt=sun_alt, sun_az=sun_az, width=width),
            width=width,
        ))
    if labels:
        layers.append(figure_labels(
            alt=alt, az=az, limit=limit, width=width, height=height,
        ))
    layers.append(compass(width=width, height=height))

    disc = mp.dstack(*layers)
    title = mp.center(mp.text(f"the sky over {place}"), width=disc.width)
    return title / disc / caption(
        when=when, jd=jd, longitude=longitude, sun_alt=sun_alt,
        width=disc.width,
    )


def disc_strokes(series: list, width: int) -> mp.plot:
    """Line series drawn in the disc's own window.

    Every stroked or dotted layer is drawn in the same window -- the square
    from `-REACH` to `REACH` on both axes -- so that `dstack` lays them over
    the backdrop with every coordinate agreeing.
    """
    return mp.line(
        *series,
        xrange=(-REACH, REACH),
        yrange=(-REACH, REACH),
        width=width,
        height=width // 2,
    )


def disc_points(series: list, width: int) -> mp.plot:
    """Scatter series drawn in the disc's own window."""
    return mp.scatter(
        *series,
        xrange=(-REACH, REACH),
        yrange=(-REACH, REACH),
        width=width,
        height=width // 2,
    )


def backdrop(
    width: int,
    sun_alt: float,
    sun_az: float,
    sky: np.ndarray,        # float[rgb]
) -> mp.plot:
    """The disc itself: sky inside the rim, and nothing outside it.

    The sky is one colour per frame, plus a sunrise: while the sun is near the
    horizon, the rim warms on the sun's side of the disc, brightest at the rim
    and falling away towards the zenith, which is dawn and dusk seen whole.
    """
    axis = np.linspace(-REACH, REACH, width)
    x, y = np.meshgrid(axis, -axis)
    radius = np.hypot(x, y)

    ground = np.ones((width, width, 1)) * sky
    # The pixel's compass bearing, east on the left, against the sun's.
    bearing = np.degrees(np.arctan2(-x, y))
    toward_sun = np.cos(np.radians(bearing - sun_az))
    burn = (
        np.exp(-((sun_alt + 1.5) / 7.0) ** 2)
        * smoothstep(0.45, 1.0, radius)
        * np.clip(toward_sun, 0.0, 1.0) ** 2.2
    )
    ground = blend(ground, GLOW, 0.85 * burn[..., None])

    # Past the rim there is no sky at all. The edge is blended across one
    # pixel rather than cut, or a bright day would show its stair-steps.
    pixel = 2 * REACH / width
    coverage = 1.0 - smoothstep(1.0 - 0.7 * pixel, 1.0 + 0.7 * pixel, radius)
    space = np.ones((width, width, 1)) * SPACE
    ground = blend(space, ground, coverage[..., None])
    return mp.image(ground.astype(np.uint8))


def horizon(width: int) -> mp.plot:
    """The rim itself: a braille circle at radius one.

    Drawn as a stroke rather than into the backdrop because braille dots are
    twice as fine as the backdrop's pixels each way, which is the difference
    between a circle and a flight of stairs.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, 4 * width)
    return disc_strokes([
        (np.cos(angle), np.sin(angle), tuple(HORIZON_RING)),
    ], width=width)


def star_series(
    alt: np.ndarray,        # float[n]
    az: np.ndarray,         # float[n]
    mag: np.ndarray,        # float[n]
    bv: np.ndarray,         # float[n]
    limit: float,
    sky: np.ndarray,        # float[rgb]
    width: int,
) -> list | None:
    """Every star the sky lets through, tinted, dimmed, and placed.

    Brightness runs linearly down the five magnitudes above the current
    limit, which is how a paper chart grades its dot sizes; a star fades all
    the way to the sky's own colour as it reaches the limit, so twilight takes
    the faint ones smoothly rather than switching them off. The few stars
    brighter than magnitude 0.2 get four extra dots, a sparkle standing in
    for the size a paper chart would spend on them.
    """
    up = (alt > 0.0) & (mag < limit)
    if not up.any():
        return None
    x, y = to_disc(alt[up], az[up])
    value = np.clip((limit - mag[up]) / 5.0, 0.0, 1.0) ** 0.8
    tint = ramp(TINT_BV, TINT_COLOR, bv[up])
    color = blend(np.ones_like(tint) * sky, tint, value[:, None])

    series = [(x, y, color.astype(np.uint8))]
    bright = mag[up] < 0.2
    if bright.any():
        dot = REACH / width          # one braille dot, in disc coordinates
        glint = blend(
            np.ones_like(tint[bright]) * sky,
            tint[bright],
            0.55 * value[bright, None],
        ).astype(np.uint8)
        for dx, dy in ((dot, 0), (-dot, 0), (0, dot), (0, -dot)):
            series.append((x[bright] + dx, y[bright] + dy, glint))
    return series


def figure_series(
    alt: np.ndarray,        # float[n]
    az: np.ndarray,         # float[n]
    limit: float,
    sky: np.ndarray,        # float[rgb]
) -> list | None:
    """The constellation figures, as one stroke per polyline.

    The figures are imaginary, so they go before the stars do: they fade with
    the limiting magnitude and are gone entirely by civil twilight. A vertex
    below the horizon becomes a break in its stroke, so a figure half-set is
    drawn only down to the rim.
    """
    presence = np.clip((limit - FIGURE_MAGNITUDE) / 1.6, 0.0, 1.0)
    if presence <= 0.0:
        return None
    color = tuple(blend(sky, FIGURE, presence).astype(np.uint8))
    x, y = to_disc(alt, az)
    x, y = x.copy(), y.copy()
    x[alt <= 0.0] = np.nan
    y[alt <= 0.0] = np.nan
    series = []
    for polylines in FIGURE_TABLE.values():
        for indices in polylines:
            if np.isfinite(x[indices]).sum() >= 2:
                series.append((x[indices], y[indices], color))
    if not series:
        return None
    return series


def sun_series(sun_alt: float, sun_az: float, width: int) -> list:
    """The sun, as a filled disc of dots about its position."""
    x, y = to_disc(np.array([sun_alt]), np.array([sun_az]))
    dot = REACH / width
    span = np.arange(-3, 4) * dot
    dx, dy = np.meshgrid(span, span)
    inside = np.hypot(dx, dy) <= 2.6 * dot
    return [(
        float(x[0]) + dx[inside],
        float(y[0]) + dy[inside],
        tuple(SUN_DISC),
    )]


def figure_labels(
    alt: np.ndarray,        # float[n]
    az: np.ndarray,         # float[n]
    limit: float,
    width: int,
    height: int,
) -> mp.plot:
    """Each constellation's tag, at the middle of what shows of its figure.

    Only figures with at least three stars up are named, and only while the
    stars themselves are out: past civil twilight there is nothing left to
    label.
    """
    if limit < FIGURE_MAGNITUDE:
        return mp.text("")
    x, y = to_disc(alt, az)
    marks = []
    for name, polylines in FIGURE_TABLE.items():
        indices = sorted({i for polyline in polylines for i in polyline})
        up = alt[indices] > 0.0
        if up.sum() < 3:
            continue
        mid_x = float(np.mean(x[indices][up]))
        mid_y = float(np.mean(y[indices][up]))
        if np.hypot(mid_x, mid_y) < 0.95:
            marks.append((mid_x, mid_y, name))
    return scribe(marks, width=width, height=height, fgcolor=LABEL)


def compass(width: int, height: int) -> mp.plot:
    """The cardinal points, on the rim where the sky meets them."""
    return scribe(
        [(0.0, 1.06, "N"), (0.0, -1.06, "S"),
         (-1.06, 0.0, "E"), (1.06, 0.0, "W")],
        width=width,
        height=height,
        fgcolor=COMPASS,
    )


def caption(
    when: datetime.datetime,
    jd: float,
    longitude: float,
    sun_alt: float,
    width: int,
) -> mp.plot:
    """The line under the disc: clock time, sidereal time, and the light."""
    if sun_alt >= SUNRISE_ALTITUDE:
        light = "day"
    elif sun_alt >= -6.0:
        light = "civil twilight"
    elif sun_alt >= -12.0:
        light = "nautical twilight"
    elif sun_alt >= -18.0:
        light = "astronomical twilight"
    else:
        light = "night"
    sidereal = (gmst_degrees(jd) + longitude) % 360.0 / 15.0
    text = (
        f"{when:%d %b %H:%M %Z} · lst"
        f" {int(sidereal):02d}:{int(sidereal % 1.0 * 60):02d} · {light}"
    )
    return mp.center(mp.text(text), width=width)


def scribe(
    marks: list[tuple[float, float, str]],
    width: int,
    height: int,
    fgcolor,
) -> mp.plot:
    """Words at positions in the disc, as one transparent text layer.

    A text plot's spaces are transparent to `dstack`, so a canvas of spaces
    with words written into it puts each word over the disc where it belongs.
    """
    grid = [[" "] * width for _ in range(height)]
    for x, y, word in marks:
        column = round((x + REACH) / (2 * REACH) * width - len(word) / 2)
        row = round((REACH - y) / (2 * REACH) * height - 0.5)
        column = min(max(column, 0), width - len(word))
        row = min(max(row, 0), height - 1)
        # Words are placed in the order given, and a word whose spot is
        # already taken tries the row below and the row above before giving
        # way: a missing label loses less than two written over each other.
        for candidate in (row, row + 1, row - 1):
            if not 0 <= candidate < height:
                continue
            reach = grid[candidate][max(column - 1, 0):column + len(word) + 1]
            if all(cell == " " for cell in reach):
                grid[candidate][column:column + len(word)] = word
                break
    return mp.text(
        "\n".join("".join(line) for line in grid),
        fgcolor=fgcolor,
    )


# # #
# Where everything is: the astronomy.


def julian_date(when: datetime.datetime) -> float:
    """The astronomer's timestamp: days, fractional, since 4713 BC."""
    return when.timestamp() / 86400.0 + 2440587.5


def gmst_degrees(jd: float) -> float:
    """Greenwich mean sidereal time: where the stars stand over Greenwich.

    Sidereal time is the right ascension crossing the meridian at this moment,
    which is all a planisphere needs to know. It gains on the clock by four
    minutes a day, which is the sun falling behind the stars as the Earth goes
    round it.
    """
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0


def sun_radec(jd: float) -> tuple[float, float]:
    """The sun's equatorial position, to a hundredth of a degree.

    The standard low-precision solar model: mean longitude, corrected by the
    equation of centre for the Earth's elliptical orbit, then tilted from the
    ecliptic onto the equator. Checked against the almanac: it puts sunset in
    Oxford within a few minutes.
    """
    n = jd - 2451545.0
    mean_longitude = np.radians((280.460 + 0.9856474 * n) % 360.0)
    mean_anomaly = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = (
        mean_longitude
        + np.radians(1.915) * np.sin(mean_anomaly)
        + np.radians(0.020) * np.sin(2 * mean_anomaly)
    )
    obliquity = np.radians(23.439 - 0.0000004 * n)
    ra = np.degrees(np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude),
        np.cos(ecliptic_longitude),
    )) % 360.0
    dec = np.degrees(np.arcsin(
        np.sin(obliquity) * np.sin(ecliptic_longitude),
    ))
    return float(ra), float(dec)


def altaz(
    ra_deg: np.ndarray,     # float[n]
    dec_deg: np.ndarray,    # float[n]
    jd: float,
    lat_deg: float,
    lon_deg: float,
) -> tuple[np.ndarray, np.ndarray]:     # float[n], float[n]
    """Equatorial coordinates brought down to earth: altitude and azimuth.

    The hour angle is how far past the meridian a star has turned, sidereal
    time minus right ascension; the rest is the spherical triangle between the
    star, the zenith and the pole. Azimuth runs from north through east.
    """
    lst = np.radians(gmst_degrees(jd) + lon_deg)
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    lat = np.radians(lat_deg)
    hour = lst - ra
    alt = np.arcsin(
        np.sin(lat) * np.sin(dec)
        + np.cos(lat) * np.cos(dec) * np.cos(hour)
    )
    az = np.arctan2(
        -np.cos(dec) * np.sin(hour),
        np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(hour),
    )
    return np.degrees(alt), np.degrees(az) % 360.0


def to_disc(
    alt: np.ndarray,        # float[n], degrees
    az: np.ndarray,         # float[n], degrees
) -> tuple[np.ndarray, np.ndarray]:     # float[n], float[n]
    """The stereographic projection: the zenith at centre, the horizon at one.

    Projecting from the nadir gives radius tan(z/2) at zenith distance z,
    which is 0 overhead and exactly 1 at the horizon. North is up and east is
    *left*, because the chart is the sky seen from underneath.
    """
    zenith_distance = np.radians(90.0 - alt)
    r = np.tan(zenith_distance / 2.0)
    azimuth = np.radians(az)
    return -r * np.sin(azimuth), r * np.cos(azimuth)


# # #
# Unpacking the catalogue.


def unpack_stars(
    packed: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The catalogue, as arrays of right ascension, declination, magnitude,
    and B-V colour index. See the note above `STARS` for the record format."""
    records = "".join(packed.split())
    codes = np.array([
        [
            int(records[i:i + 3], 16),
            int(records[i + 3:i + 6], 16),
            int(records[i + 6:i + 8], 16),
            int(records[i + 8:i + 10], 16),
        ]
        for i in range(0, len(records), 10)
    ], dtype=float)
    return (
        codes[:, 0] / 10.0,
        codes[:, 1] / 10.0 - 90.0,
        codes[:, 2] / 20.0 - 1.5,
        codes[:, 3] / 20.0 - 0.5,
    )


def unpack_figures(packed: str) -> dict[str, list[np.ndarray]]:
    """Each constellation's polylines, as arrays of star indices.

    A constellation may spread over several lines of the table, each repeating
    its tag; the polylines accumulate. See the note above `FIGURES`.
    """
    figures: dict[str, list[np.ndarray]] = {}
    for line in packed.splitlines():
        name, polylines = line.split(" ")
        figures.setdefault(name, []).extend(
            np.array([int(v, 16) for v in polyline.split("-")])
            for polyline in polylines.split(",")
        )
    return figures


STAR_TABLE = unpack_stars(STARS)
FIGURE_TABLE = unpack_figures(FIGURES)


# # #
# Small colour arithmetic, shared by the layers.


def ramp(
    stops: np.ndarray,      # float[stops]
    colors: np.ndarray,     # float[stops, rgb]
    at: np.ndarray,         # float[...]
) -> np.ndarray:            # float[..., rgb]
    """Read a colour ramp, interpolating each channel between its stops."""
    return np.stack([
        np.interp(at, stops, colors[:, channel]) for channel in range(3)
    ], axis=-1)


def blend(under: np.ndarray, over: np.ndarray, alpha) -> np.ndarray:
    """`over` laid on `under`, `alpha` of the way."""
    return under * (1 - alpha) + over * alpha


def smoothstep(low: float, high: float, at: np.ndarray) -> np.ndarray:
    """Zero below `low`, one above `high`, and an S between them."""
    t = np.clip((at - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


if __name__ == "__main__":
    tyro.cli(main)
