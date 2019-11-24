import pandas as pd
import logging
import random
logger = logging.getLogger(__name__)

TEXTCOL = 'text'
LABELCOL = 'event'
# Event codes -> model class label (unused)
EVENTMAP = {}
# Model class label -> event code (unused)
PREDICTMAP = {}

ABBREV_MAP = { r'\bINJ\b': 'INJURY',
               r'\bOBS\b': "OBSERVATION",
               r'\bCO\b': 'COMPLAINS OF',
               r'\bC O\b': 'COMPLAINS OF',
               r'\bSUS\b': 'SUSTAINED',
               r"\bSUST\b": "SUSTAINED",
               r'\bLT\b': 'LEFT',
               r'\bRT\b': 'RIGHT',
               r'\bL\b': 'LEFT',
               r'\bR\b': 'RIGHT',
               r'\bY\s?O\s?F\b': 'YEAR OLD FEMALE',
               r'\bYOWF\b': 'YEAR OLD WHITE FEMALE',
               r"\bYOWM\b": "YEAR OLD WHITE MALE",
               r"\bYOBF\b": "YEAR OLD BLACK FEMALE",
               r"\bYOHF\b": "YEAR OLD HISPANIC FEMALE",
               r"\bYOBM\b": "YEAR OLD BLACK MALE",
               r"\bY\s?O\s?M\b": "YEAR OLD MALE",
               r"\bYM\b": "YEAR OLD MALE",
               r"\bYF\b": "YEAR OLD FEMALE",
               r'\bM\b': 'MALE',
               r'\bF\b': 'FEMALE',
               r'\bY\s?O\b': "YEAR OLD",
               r'\bPT\b': 'PATIENT',
               r"\bPTS\b": "PATIENTS",
#               ' WK ': ' WEEK ',
#               ' WKS ': ' WEEKS ',
#               ' LB ': ' POUND ',
#               ' HISP ': ' HISPANIC ',
#               ' HR ': ' HOUR ',
               r'\bLAC\b': 'LACERATIONS',
               r'\bDON T\b': "DO NOT",
               r'\bDIDN T\b': "DID NOT",
               r"\bDOESN T\b" : "DOES NOT",
               r'\bIV\b': 'INTRAVENOUS',
#               ' DX ': ', DIAGNOSIS: ',
#               ' D X ': ', DIAGNOSIS: ',
               r'\bD\s?X\b': 'DIAGNOSIS',
               r'\bMVC\b': 'MOTOR VEHICLE COLLISION',
               r'\bW\b': 'WITH',
               r'\bPTA\b': 'PRIOR TO ARRIVAL',
               r'\bGSW\b': 'GUN SHOT WOUND',
               r'\bC\s?H\s?I\b': 'CLOSED HEAD INJURY',
               r'\bL\s?O\s?C\b': "LOSS OF CONSCIOUSNESS",
               r"\bX\b": "FOR",
               r"\bFX\b": "FRACTURE",
               r"\bBIBEMS\b": "BROUGHT IN BY EMERGENCY MEDICAL SERVICES",
               r"\bCLSD\b": "CLOSED",
               r"\bSTS\b": "SAYS THAT SHE",
               r"\bSTH\b": "SAYS THAT HE",
               r"\bEMS\b": "EMERGENCY MEDICAL SERVICES",
#               r"\bP\b": "PATIENT",
               r"\bPNS\b": "PAINS",
               r"\bHD\b": "HEAD",
               r"\bS\s?P\b": "STATUS POST",
               r"\bETOH\b": "ETHANOL",
               r"\bPW\b": "PUNCTURE WOUND",
               r"\bW.\b": "WITH",
               r"\bLBP\b": "LOW BLOOD PRESSURE",
               r"\bACC\b": "ACCIDENT",
               r'\bHTN\b': 'HYPERTENSION',
               r'\b2 2\b': 'SECONDARY TO',
               r'\bAB\b': 'ABRASION',
               r'\bNEC\b|\bN\.E\.C\.?\b': 'NOT ELSEWHERE CLASSIFIED',
               r'\bH\s?X\b': 'HISTORY',
               r'\bP\s?W\b': "PRESENTS WITH",
               r'\bO2\b': "WATER",
               r'\bPPL\b': "PEOPLE",
               r'2x4': "TWO BY FOUR",
#               r'\d+': '',
            }

EVENTCSV = """
name,event,desc,unknown01,unknown02
Event,10,"Violence and other injuries by persons or animals, unspecified",2,2
Event,11,Intentional injury by person,2,3
Event,12,Injury by person--unintentional or intent unknown,2,25
Event,13,Animal and insect related incidents ,2,44
Event,20,"Transportation incident, unspecified",2,65
Event,21,Aircraft incidents,2,66
Event,22,Rail vehicle incidents,2,87
Event,23,Animal and other non-motorized vehicle transportation incidents,2,101
Event,24,Pedestrian vehicular incidents,2,119
Event,25,Water vehicle incidents,2,146
Event,26,Roadway incidents involving motorized land vehicle,2,161
Event,27,Nonroadway incidents involving motorized land vehicles ,2,189
Event,29,"Transportation incident, n.e.c.",2,213
Event,30,"Fire or explosion, unspecified ",2,215
Event,31,Fires,2,216
Event,32,Explosions,2,226
Event,40,"Fall, slip, trip, unspecified ",2,234
Event,41,Slip or trip without fall,2,235
Event,42,Falls on same level ,2,251
Event,43,Falls to lower level,2,264
Event,44,Jumps to lower level,2,293
Event,45,Fall or jump curtailed by personal fall arrest system ,2,313
Event,49,"Fall, slip, trip, n.e.c.",2,314
Event,50,"Exposure to harmful substances or environments, unspecified",2,316
Event,51,Exposure to electricity,2,317
Event,52,Exposure to radiation and noise,2,327
Event,53,Exposure to temperature extremes,2,339
Event,54,Exposure to air and water pressure change,2,345
Event,55,Exposure to other harmful substances,2,349
Event,56,"Exposure to oxygen deficiency, n.e.c.",2,367
Event,57,"Exposure to traumatic or stressful event, n.e.c. ",2,373
Event,59,"Exposure to harmful substances or environments, n.e.c.",2,374
Event,60,"Contact with objects and equipment, unspecified",2,376
Event,61,Needlestick without exposure to harmful substance,2,377
Event,62,Struck by object or equipment,2,378
Event,63,Struck against object or equipment,2,418
Event,64,Caught in or compressed by equipment or objects,2,431
Event,65,"Struck, caught, or crushed in collapsing structure, equipment, or material",2,442
Event,66,Rubbed or abraded by friction or pressure,2,451
Event,67,"Rubbed, abraded, or jarred by vibration",2,458
Event,69,"Contact with objects and equipment, n.e.c.",2,463
Event,70,"Overexertion and bodily reaction, unspecified",2,465
Event,71,Overexertion involving outside sources,2,466
Event,72,Repetitive motions involving microtasks,2,490
Event,73,Other exertions or bodily reactions,2,498
Event,74,"Bodily conditions, n.e.c.",2,534
Event,78,Multiple types of overexertions and bodily reactions,2,535
Event,79,"Overexertion and bodily reaction and exertion, n.e.c.",2,536
Event,99,Unclassifiable,2,600"""

try:
    with open("augment.txt") as f:
        texts = []
        sexes = []
        ages = []
        events = []
        for line in f:
            texts.append(line)
            sexes.append(random.randint(1,2))
            ages.append(random.randint(3,100))
            events.append(99)
    AUGMENT_DF = pd.DataFrame({ TEXTCOL: texts, 'age': ages, 'sex': sexes, LABELCOL: events })
except Exception as exc:
    logger.warning("{}".format(exc))
    AUGMENT_DF = pd.DataFrame()

maxidx = 0
for line in EVENTCSV.split("\n")[2:]:
    try:
        event = line.split(',')[1]
    except:
        continue
    if not EVENTMAP.get(int(event)):
        EVENTMAP[int(event)] = maxidx
        PREDICTMAP[maxidx] = int(event)
        maxidx += 1
assert len(EVENTMAP) == 49, "malformed EVENTMAP (len == %d)" % len(EVENTMAP)
