from src.engine.BaseArena import BaseArena
import random
from typing import List, Tuple
class Manual(BaseArena):
    """
    This class allows you to send literal training data
    for example if you need repeatable results
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[float, float]]:
        return [(33.899692251686986, 1.4561135735746422, 124611.30390221025), (28.87491315777392, 5.123310709065132, 116871.36089145203), (5.9278801793919955, 1.1569221904797837, 40097.48491913555), (36.41616435792206, 7.912531546678073, 145073.5561671223), (14.202421662687858, 5.798377445638935, 74204.01987934145), (17.84469661213444, 7.369122129729218, 88272.33409586176), (32.22352009556543, 6.015811696862093, 128702.18368042048), (13.788497254857717, 7.982183863322198, 77329.85949121755), (35.726432910393086, 1.9032704645903316, 130985.83966035991), (25.856725542626297, 0.37538562806713305, 98320.94788401315), (13.962938256368837, 4.259579322966821, 70407.97341504015), (17.237780291198256, 2.6475450289215017, 77008.43093143778), (8.570429157867338, 2.2551986151574805, 50221.68470391698), (22.289431408695645, 6.415475925766168, 99699.24607761926), (32.18149183553844, 1.822817829956347, 120190.111166528), (30.939031336540847, 3.006836115732611, 118830.76624108775), (3.879497855043361, 3.14490092971986, 37928.295424569806), (8.930027051513374, 6.4518913224153, 59693.86379937072), (13.032130870102412, 0.9879089038327757, 61072.21041797279), (37.77192153141729, 7.741263795644171, 148798.29218554022), (2.080893785079976, 0.9487385811383939, 28140.158517516713), (13.740856890379657, 4.267714538036352, 69757.99974721168), (30.788903413244046, 6.795589628132469, 125957.88949599708), (14.599977282507602, 2.4837345056094895, 68767.40085874179), (32.87373958262983, 7.820279425141529, 134261.77759817254), (27.707030165289183, 0.6762777068175136, 104473.64590950258), (19.557476284904148, 7.564485243746474, 93801.39934220539), (17.605046472625666, 6.873899305613315, 86562.93802910362), (30.841155079863178, 0.4268375452299731, 113377.14033004949), (2.350338364047433, 5.632116843232296, 38315.24877860689), (10.821694933049372, 3.3924480993359047, 59249.98099781992), (26.171648202121993, 2.706611657690412, 103928.1679217468), (4.253998055155712, 7.431243515542093, 47624.48119655132), (31.984945471950073, 5.013472647518873, 125981.78171088797), (14.639454398828553, 0.17617894423948588, 64270.721084964636), (20.828830563538745, 7.8381298978633405, 98162.75148634292), (7.311047991263115, 7.08068139825915, 56094.50677030765), (10.951921608675272, 0.7412261067614665, 54338.21703954875), (35.6553423590892, 7.977622548391297, 142921.2721740502), (0.7054949104587926, 2.9829545798342556, 28082.39389104489), (21.472160435839996, 1.34144100846419, 87099.36332444836), (38.39737671277089, 5.302808795616582, 145797.74772954584), (33.40301682194682, 1.1045767547367307, 122418.20397531393), (15.53411303225474, 1.1603177283837631, 68922.97455353175), (32.74560423195594, 4.788413982728023, 127813.64066132387), (38.78729731691753, 0.4037259246219529, 137169.34379999648), (1.4351520001368456, 1.2160426808541684, 26737.541362118875), (4.068095390320807, 6.865850663056709, 45935.98749707584), (39.55579229297766, 0.3105406652951075, 139288.4582095232), (16.591196236114346, 2.1319728567446763, 74037.53442183239), (30.534139336195267, 6.00777786380527, 123617.97373619633), (17.656416310023584, 7.406108891201315, 87781.46671247338), (27.14435971804797, 7.672274616194784, 116777.62838653347), (27.088805286765737, 7.725209574808976, 116716.83500991517), (11.351604933462283, 2.675146396023557, 59405.10759243396), (33.61124070794572, 1.1696992133886468, 123173.12055061445), (13.381812629973581, 5.25868554771901, 70662.80898535876), (32.810776620465376, 0.07413588007744476, 118580.60162155102), (35.92750376700087, 7.5855173244549015, 142953.54594991243), (9.58499518509523, 7.850463369570138, 64455.912294425965), (5.657431404254565, 0.7512592914101113, 38474.812795583915), (22.535957703749126, 2.3025111577345108, 92212.8954267164), (34.43113546563215, 3.899970838448249, 131093.34807379296), (10.1103144524577, 5.841772232453172, 62014.48782227944), (29.98285319792261, 4.3432473276621435, 118635.05424909212), (20.81680906138297, 7.122224958411719, 96694.87710097234), (38.21778870914139, 2.393985382691863, 139441.3368928079), (38.22154530197655, 6.194750516674852, 147054.13693927933), (4.1550279796385725, 7.865017815845925, 48195.11957060757), (28.83143709360661, 1.6970084254366355, 109888.32813169311), (20.787621156958217, 2.638114166431844, 87639.09180373832), (18.00974707408583, 1.135803108960265, 76300.84744017803), (4.537507299979016, 7.86582321768194, 49344.16833530093), (12.166916107767843, 2.561901229717364, 61624.55078273826), (13.093032843213685, 5.620166698759594, 70519.43192716024), (17.46077647267041, 2.729501753439127, 77841.33292488949), (31.888307889283848, 7.279399739942035, 130223.72314773561), (28.228433043308733, 6.236834035939108, 117158.96720180442), (0.9428344941930922, 3.064383405414323, 28957.27029340792), (38.77711390094257, 0.4747798033176691, 137280.90130946305), (5.866733410422542, 2.066392543111112, 41732.98531748985), (3.8293704301328946, 5.241653728866962, 41971.41874813261), (20.815824403035656, 3.253142241694225, 88953.75769249542), (14.770462188008766, 6.269287383952722, 76849.96133193174), (13.677656194474386, 7.97847248081647, 76989.9135450561), (18.340017347442547, 1.5239361866729153, 78067.92441567348), (29.65794180878746, 6.656468094612958, 122286.7616155883), (23.704677186045792, 3.5450423744265915, 98204.11630699056), (17.9239652995834, 6.011393803039764, 85794.68350482972), (37.70745021384171, 7.918506713332163, 148959.36406818943), (17.118000145158184, 0.39631704149679337, 72146.63451846814), (12.381694124559317, 2.856693264837534, 62858.468903353016), (22.505435123411125, 3.7620682340484093, 95040.44183833018), (28.86376896835918, 6.210693307038275, 119012.69351915408), (28.43107747698001, 2.525611926336982, 110344.456283614), (14.186973712474593, 3.2786553770093025, 69118.23189144238), (17.88404769842633, 2.235728840015109, 78123.60077530921), (13.241325888905653, 3.1087587672006336, 65941.49520111822), (2.528553566972618, 5.7114896492602005, 39008.63999943825), (23.027879779751437, 3.612056088012481, 96307.75151527928)]

#        return [(3.0829800228956428, 4.48830093538644, 30.780635057213185), (19.394768240791976, 4.132484554096511, 99.9506658661515), (27.781448489434467, 7.937430197901479, 138.2126528290548), (33.62324173819261, 0.9167418587108083, 159.68853059040225), (13.44647911402188, 6.397024700905064, 59.601935034159176), (22.203044344485036, 5.940330467735012, 123.77976441327952), (5.095217969013821, 1.8145516517925921, 42.4998581701376), (37.31584689129786, 1.609365422866146, 205.82347798936567), (1.5668165888473817, 4.609630264389032, 56.22216369655404), (22.4859759837376, 0.8953329533927405, 130.9537892674283)]


