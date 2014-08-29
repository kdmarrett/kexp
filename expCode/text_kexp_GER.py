# text_kexp.py
# Author: Karl Marrett
# Text and other related dictionaries for runKexp.py script

import numpy as np

cont_btn = 8
cont_btn_label = 'Next'

wait_brief = .2
wait_long = 2

# INSTRUCTIONS AND FEEDBACK
instr = dict()
button_keys = dict()

# SECTION 1 (INTRODUCTION) INSTRUCTIONS
instr['s0_start_sect'] = ('In diesem Experiment werden Sie Buchstaben h\"oren, die in 3 r\"aumliche Gruppen eingeteilt sind'
                          '(Links, Mitte, Rechts)'
                          'Jede Buchstabengruppe hat einen eigenen Sprecher.'
                          'Am Anfang jedes Trials werden sie mit der Gruppierung der Buchstaben vertraut gemacht. Hierbei werden die Gruppen entsprechend ihre r\"aumlichen Orientierung'
                          'visuell auf dem Bildschirm angezeigt.'
                          'Sie werden feststellen, dass einer der visuell angezeigten Buchstaben gr\"un gef\"arbt ist, und laut vorgesprochen wird.'
                          'Dies ist ihr Zielbuchstabe f\"ur diesen Trial.'
                          'Die Gruppen enthalten das gesamte Alphabet und zus\"atzlich die gesprochenen Anweisungen \'Lesen\', \'Pause\',\'Leerzeichen\' und \'L\"oschen\'.'
                          'Wir bitten Sie sich w\"ahrend eines Trials auf Ihr Zielwort zu konzentrieren.'
                          'Da wir ihre Augenbewegungen aufnehmen, bitten wir Sie w\"ahrend des Trials mit ihren Augen das Fixationskreuz zu fokussieren.'
                          'Dr\"ucken Sie "{}" um fortzufahren' .format(cont_btn_label))


button_keys['s0_start_sect'] = [cont_btn]
instr['s0_start_block_0'] = ('Im ersten Teil werden wir Sie mit verschiedenen Konditionen des Experiments vertraut machen.'
                              'Er beinhaltet 1 Block f\"ur jede Kondition, wobei es 8 verschiedene Konditionen gibt.'
                              'Bitte konzentrieren Sie sich die ganze Zeit auf den gr\"un gef\"arbten Buchstaben und ignorieren Sie alle anderen.'
                              'Um die Aufgabe schwieriger zu machen, werden Sie auch verschiedene Hintergrundger\"ausche h\"oren.'
                              'Machen Sie sich keine Sorgen falls es Ihnen zu Beginn schwer f\"allt den Buchstaben zuh\"oren.'
                              'Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))

button_keys['s0_start_block_0'] = [cont_btn]
instr['s0_start_trial_0'] = ('W\"ahrend dieser Kondition sind die W\"orter in alphabetischer Reihenfolge den Gruppen zugeteilt. Jeder Buchstabe hat zus\"atzlich einen eigenen k\"unstlichen Ton.'
                              'Sie k\"onnen die r\"aumliche Einteilung, den Sprecher und den Ton verwenden um sich auf ihren Zielbuchstaben zu konzentrieren.'
                               'Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))

button_keys['s0_start_trial_0'] = [cont_btn]

instr['s0_start_trial_1'] = ('In dieser Kondition, sind die Buchstaben in alphabetischer Reihenfolge den Gruppen zugeteilt. Zus\"atzlich hat jeder Buchstabe einen individuellen k\"unstlichen Ton.'
                             'Die Geschwindigkeit mit der die Buchstaben pr\"asentiert werden, variiert zwischen den einzelnen Gruppen.'
                             'Sie k\"onnen den Sprecher, Ort, Ton sowie die Geschwindigkeit verwenden um sich auf Ihren Zielbuchstaben zu konzentrieren.'
                             ' Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))




button_keys['s0_start_trial_1'] = [cont_btn]
instr['s0_start_trial_2'] = ('In dieser Kondition sind die Buchstaben den Gruppen und individuellen k\"unstlichen T\"onen zuf\"allig zugeteilt.'
                             'Trotz der zuf\"alligen Einteilung hat jeder Buchstabe einen festen Ton w\"ahrend des gesamten Trials und die T\"one werden in gleichm\"assigem Muster zugeteilt..'
                             ' Sie k\"onnen den Sprecher, Ton und die r\"aumliche Orientierung ihres Zielbuchstaben verwenden um sich darauf zu konzentrieren.'
                             ' Dr\"ucken Sie "{}" um zu starten.'.format(cont_btn_label))


button_keys['s0_start_trial_2'] = [cont_btn]
instr['s0_start_trial_3'] = ('In dieser Kondition sind die Buchstaben in alphabetischer Reihenfolge und mit individuellem Ton in die Gruppen eingeteilt.'
                             'Der Zielbuchstabe in dieser Kondition ist immer einer der Folgenden: \'B\', \'C\', \'D\', \'E\', \'G\', \'P\', \'T\','
                             ' \'V\', or \'Z\'. Sie k\"onnen den Sprecher, Ton und die r\"aumliche Orientierung benutzen um sich auf den Zielbuchstaben zu konzentrieren.'
                             ' Dr\"ucken Sie "{}" um zu starten.'.format(cont_btn_label))

i
button_keys['s0_start_trial_3'] = [cont_btn]
instr['s0_start_trial_4'] = ('In dieser Kondition sind die Buchstaben in alphabetischer Ordnung den Gruppen zugeteilt.'
                             ' Den Buchstaben ist jedoch kein individueller Ton zugeordnet.'
                             ' Sie k\"onnen daher die Richtung und den Sprecher des Buchstabens verwenden um sich darauf zu konzentrieren aber nicht den Ton.'
                             ' Dr\"ucken Sie "{}" um zu starten.'.format(cont_btn_label))


button_keys['s0_start_trial_4'] = [cont_btn]
instr['s0_start_trial_5'] = ('In dieser Kondition sind die Buchstaben in alphabetische Reihenfolge den Gruppen und individuellen T\"onen zugeteilt.'
                             ' Jeder Buchstabe bekommt einen Ton in zuf\"alliger Reihenfolge zugeteilt. Dadurch ergibt sich kein Muster auf das Sie sich w\"ahrend des Trials verlassen k\"onnen.'                             'so you can get not rely on the any particular pattern of tones to help guide you. '
                             'Sie k\"onnen den Sprecher, die r\"aumliche Orientierung und den Ton des Zielbuchstabens verwenden um sich darauf zu konzentrieren.'
                             ' Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))


button_keys['s0_start_trial_5'] = [cont_btn]
instr['s0_start_trial_6'] = ('In dieser Kondition sind die Buchstaben den Gruppen und T\"onen in alphabetischer Reihenfolge zugeordnet. '
                             'Die Lautst\"arke der einzelnen Sprecher Oszilliert in verschiedenen Geschwindigkeiten.'
                             'Daher k\"onnen sie den Sprecher, die r\"aumliche Orientierung, den Ton und die Ver\"anderung der Lautst\"arke verwenden um sich auf ihren Zielbuchstaben zu konzentrieren.'
                             'Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))

button_keys['s0_start_trial_6'] = [cont_btn]
instr['s0_start_trial_7'] = ('In dieser Kondition, sind die Buchstaben in alphabetischer Reihenfolge den Gruppen und T\"onen zugeordnet.'
                             'Die Lautst\"arke der Sprecher wird mit der gleichen Geschwindigkeit oszillieren.'
                             ' Sie k\"onnen den Ort, Ton und Sprecher, sowie zus\"atzlich die unterschiedliche Lautst\"arke verwenden um sich auf Ihren Zielbuchstaben zu konzentrieren.'
                             ' Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))


button_keys['s0_start_trial_7'] = [cont_btn]
instr['s0_end_trial'] = (
    'Sehr gut! Sie sind fertig mit diesem trial.'.format(cont_btn_label))

button_keys['s0_end_trial'] = []
instr['s0_end_block'] = (
    'Sie sind fertig mit dem Block dieses Teils.'.format(cont_btn_label))

button_keys['s0_end_block'] = []
instr['s0_end_sect'] = ('Sehr gut! Sie sind fertig mit dem ersten Teil des Experiments. Sie k\"onnen eine Pause machen wenn Sie m\"ochten.'
      ' Dr\"ucken Sie "{}" um weiter zu machen.'.format(cont_btn_label))


button_keys['s0_end_sect'] = [cont_btn]



# SECTION 2 INSTRUCTIONS
instr['s1_start_sect'] = ('Sie beginnen jetzt den zweiten Teil des Experiments. In diesem Teil gibt es 5 Bl\"ocke mit jeweils 9 Trials.'
                          'Die einzelnen Trials folgen direkt aufeinander. Die Trials bestehen aus den verschiedenen Konditionen in gemischter Reihenfolge.'
                          'Es wird keine Anweisungen zu Beginn eines Trials geben. Bitte benutzen Sie die visuelle Anzeige oder den auditorischen Hinweis zu Beginn des Trials um die entsprechende Kondition herauszufinden.'
                          ' Wenn Sie bereit sind anzufangen, dr\"ucken Sie "{}"'.format(cont_btn_label))


button_keys['s1_start_sect'] = [cont_btn]
instr['s1_start_block_0'] = (
    'Sie beginnen mit dem ersten Block dieses Teils des Experiments. Los geht\'s!'.format(cont_btn_label))

button_keys['s1_start_block_0'] = []

instr['s1_start_trial_0'] = ('')

instr['s1_start_trial_1'] = ('')

instr['s1_start_trial_2'] = ('')

instr['s1_start_trial_3'] = ('')

instr['s1_start_trial_4'] = ('')

instr['s1_start_trial_5'] = ('')

instr['s1_start_trial_6'] = ('')

instr['s1_start_trial_7'] = ('')

instr['s1_start_trial_8'] = ('')

button_keys['s1_start_trial_0'] = []

instr['s1_start_block_1'] = ('Es folgt der zweite Block dieses Teils des Experiments. '
                             'Zur Erinnerung: ein neuer Trial startet immer kurz nach dem der letzte beendet wurde.'
                             'Es gibt 8 Trials in diesem Block. Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))



button_keys['s1_start_block_1'] = [cont_btn]

instr['s1_start_block_2'] = ('Es folgt der dritte Block dieses Teils des Experiments. '
                             'Zur Erinnerung: ein neuer Trial startet immer kurz nach dem der letzte beendet wurde.'
                             'Es gibt 8 Trials in diesem Block. Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))




button_keys['s1_start_block_2'] = [cont_btn]

instr['s1_start_block_3'] = ('Es folgt der vierte Block dieses Teils des Experiments. '
                             'Zur Erinnerung: ein neuer Trial startet immer kurz nach dem der letzte beendet wurde.'
                             'Es gibt 8 Trials in diesem Block. Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))


button_keys['s1_start_block_3'] = [cont_btn]

instr['s1_start_block_4'] = ('Es folgt der letzte Block dieses Teils des Experiments. '
                             'Zur Erinnerung: ein neuer Trial startet immer kurz nach dem der letzte beendet wurde.'
                            'Es gibt 8 Trials in diesem Block. Dr\"ucken Sie "{}" um zu beginnen.'.format(cont_btn_label))


button_keys['s1_start_block_4'] = [cont_btn]


instr['s1_end_trial'] = ('')

button_keys['s1_end_trial'] = []
instr['s1_end_block'] = ('Sehr gut! Machen Sie eine Pause wenn Sie m\"ochten und dr\"ucken Sie "{}" '
                         'wenn Sie bereit f\"ur den n\"achsten Block sind.'.format(cont_btn_label))

button_keys['s1_end_block'] = [cont_btn]
instr['s1_end_sect'] = ('Gut gemacht! Sie sind fertig mit dem zweiten Teil des Experiments. Machen Sie eine'
                        'Pause wenn Sie m\"ochten!'
                        'Dr\"ucken Sie "{}" wenn Sie bereit sind weiterzumachen.'.format(cont_btn_label))

button_keys['s1_end_block'] = [cont_btn]

# SECTION 3 COGNTIVE LOAD ASSESSMENT INSTRUCTIONS
instr['s2_start_sect'] = ('Sie beginnen jetzt mit dem letzten Teil des Experiments.'
                          ' Wie bisher achten Sie bitte in jedem Trial auf ihren Zielbuchstaben.'
                          ' In diesem Teil gibt es einen Trial f\"ur jede Kondition. Nach jedem Trial werden wir Ihnen einige Fragen zur Schwierigkeit dieser Kondition stellen.'
                          ' Wir interessieren uns f\"ur Ihre Erfahrung in dem Experiment und welche Arbeitsbelastung sie dabei empfanden.'
                          ' Wir m\"ochten herausfinden welche Faktoren Sie dabei beeinflusst haben. Diese Faktoren k\"onnen von der Aufgabe kommen, oder davon wie Sie sich dabei gef\"uhlt haben, '
                          'etwa wie zufrieden Sie mit sich waren, wie viel Anstrengung Sie auf die Aufgabe gerichtet haben oder wie frustrierend Sie die Aufgabe fanden.'
                          ' Daher m\"ochten wir Sie bitten diese Faktoren zu gut wie m\"oglich zu evaluieren.'
                          ' Dr\"ucken Sie "{}" um fortzufahren.'.format(cont_btn_label))



utton_keys['s2_start_sect'] = [cont_btn]
instr['s2_start_block_0'] = ('Nach jeder Kondition,'
                             ' bitten wir Sie ihre Erfahrung im Bezug auf verschiedene Faktoren von 0 bis 9 zu bewerten.' 
                             'Bitte bewerten Sie jede Kondition unabh\"angig von den anderen und beachten Sie die Beschreibung der Bewertung gr\"undlich. '
                             'Falls Sie fragen zum Bewertungsbogen haben, z\"ogern Sie nicht uns zu fragen. Dr\"ucken Sie "{}" um fortzufahren.'.format(cont_btn_label))


button_keys['s2_start_block_0'] = [cont_btn]

# trial instr taken from section 1
#continue automatically
instr['s2_end_trial'] = ('Vielen Dank f\"ur Ihre Antworten..')


button_keys['s2_end_trial'] = []
instr['s2_end_block'] = ('Vielen Dank f\"ur Ihre Antworten. Machen Sie eine Pause wenn Sie m\"ochten und dr\"ucken Sie "{}" wenn  '
                         'Sie bereit f\"ur den n\"achsten Block sind.'.format(cont_btn_label))




button_keys['s2_end_block'] = [cont_btn]

instr['s2_end_sect'] = ('Fertig! Vielen Dank f\"ur Ihre Teilnahme!')


button_keys['s2_end_block'] = []

gen_survey = dict()
gen_survey[0] = ('Wie mental anstrengend war diese Aufgabe? Wie viel mentale und perpetuelle Aktivit\"at war gefragt (z.B. denken, '
                 ' entscheiden, erinnern, hinschauen, suchen )? War die Aufgebe leicht oder anspruchsvoll, einfach'
                 'oder komplex? Geben Sie eine Zahl von 0 bis 9 ein wobei 9 sehr anspruchsvoll bedeutet.'.format(cont_btn_label))




gen_survey[1] = ('Wie viel k\"orperliche Aktivit\"at war gefragt (z.B. dr\"ucken, ziehen, drehen, kontrollieren, aktivieren)?'
                 ' War die Aufgabe leicht oder anspruchsvoll, langsam oder schnell, entspannt oder anstrengend, gem\"utlich oder m\"uhsam? Geben Sie eine Zahl von 0 bis 9 ein, wobei 9 sehr anspruchsvoll bedeutet.'.format(cont_btn_label))


gen_survey[2] = ('Wie viel zeitlichen Druck haben Sie versp\"urt um mit der Geschwindigkeit der Aufgabe mit zu kommen?'
                 ' War die Geschwindigkeit langsam und gem\"achlich oder schnell und anspruchsvoll? Geben Sie eine Zahl von 0 bis 9 ein, wobei 9 sehr anspruchsvoll bedeutet.'.format(cont_btn_label))

gen_survey[3] = ('Wie erfolgreich haben Sie ihrer Meinung nach die Aufgabe ausgef\"uhrt? '
                 ' Wie zufrieden sind Sie mit ihrer Leistung?'
                 'Geben Sie eine Zahl von 0 bis 9 ein, wobei 9 ... bedeutet.' .format(cont_btn_label))



gen_survey[4] = ('Wie sehr haben Sie sich angestrengt (sowohl mental als auch physisch) um ihre Leistung zu erreichen?'
                 'Geben Sie eine Zahl von 0 bis 9 ein, wobei 9 ... bedeutet.'.format(cont_btn_label))



gen_survey[5] = ('Wie unsicher, entmutigt, irritiert, gestresst, genervt im Gegensatz zu sicher, ermutigt, zufrieden, entspannt haben Sie sich w\"ahrend der Aufgabe gef\"uhlt?'
                 'Geben Sie eine Zahl zwischen 0 und 9 ein, wobei 9 ... bedeutet.'.format(cont_btn_label))

gen_survey['Feedback'] = 'Sie haben ' + str(response) + ' eingegeben, falls dies richtig ist dr\"ucken Sie "{}". Dr\"ucken Sie irgendeine Taste um die Eingabe zu wiederholen.'.format(cont_btn_label)
gen_survey['AssertionError'] = "Bitte geben Sie eine Zahl zwischen " + str(min(response_btns)) + " und " + str(max(response_btns)+" ein!")
gen_survey['ValueError'] = "Bitte geben Sie eine einstellige Zahl ein."



mid_survey = dict()
mid_survey[0] = ('W\"ahrend des Experiments wurden Ihre Bewertungen verwendet um ihre Erfahrung mit den verschiedenen Konditionen zu erfassen.'
                 'Diese Bewertungen k\"onnen auf unterschiedliche Weise interpretiert werden. So haben manche Leute das Gef\"uhl, dass '
                 'mentale und zeitliche Anforderungen den Hauptteil ihrer Belastung ausmachen unabh\"angig von der Anstrengung'
                 ' die Sie aufbringen oder dem Erfolg den Sie haben. '
                 'Die Evaluierung die Sie gleich ausf\"uhren werden, wird die relative Gewichtung von sechs verschiedenen Faktoren erfassen auf ihre Belastung. '
                 ' Dr\"ucken Sie "{}" um fortzufahren.'.format(cont_btn_label))



mid_survey[1] = ('Jetzt werden wir Ihnen eine Reihe von Paaren von Begriffen zeigen (z. B. Anstrengung versus Mentale Anforderung )'
                 'Bitte w\"ahlen Sie den Begriff, der Ihrer Meinung nach gr\"osseren Einfluss auf ihre Belastung w\"ahrend der Aufgabe hatte.'
                 'W\"ahlen Sie sorgf\"altig und beachten Sie dass ihre Wahl konsistent mit Ihren Angaben w\"ahrend der Bewertung ist.'
                 'Es gibt keine richtige Antwort -- wir interessieren uns nur f\"ur ihre Meinung.'
                 'Dr\"ucken Sie "{}" um fortzufahren.'.format(cont_btn_label))


rel_survey = dict()
rel_survey[
    0] = '1. K\"orperlicher Anspruch oder 2. Zeitlicher Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'

    rel_survey[
    1] = '1. Anstrengung oder 2. Mentaler Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'

    rel_survey[
    2] = '1. Frustration oder 2. K\"orperlicher Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'

    rel_survey[
    3] = '1. Anstrengung oder 2. Frustration. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'

rel_survey[
    4] = '1. Mentaler Anspruch oder 2. Zeitlicher Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    5] = '1. K\"orperlicher Anspruch oder 2. Anstrengung. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    6] = '1. Zeitlicher Anspruch oder 2. Leistung. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    7] = '1. Frustration oder 2. Mentaler Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    8] = '1. Zeitlicher Anspruch oder 2. Frustration. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    9] = '1. Leistung oder 2. Anstrengung. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    10] = '1. Anstrengung oder 2. Zeitlicher Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    11] = '1. Frustration oder 2. Leistung. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    12] = '1. Leistung oder 2. K\"orperlicher Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    13] = '1. Mentaler Anspruch oder 2. Leistung. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'
rel_survey[
    14] = '1. Mentaler Anspruch oder 2. K\"orperlicher Anspruch. Dr\"ucken Sie die Zahl des Aspekts, der Sie Ihrer Meinung nach mehr beeinflusst hat.'


# ASSIGN BUTTON_KEYS AND WAIT TIMES BY GENERAL PATTERN
template = dict.fromkeys(instr)
#override template with explicit values so far
button_keys = dict(template.items() + button_keys.items())
for key in dict.keys(instr):
    value = instr.get(key)
    if (value == ('')):
        button_keys[key] = []

wait_keys = dict.fromkeys(button_keys)
for key in dict.keys(wait_keys):
    value = button_keys.get(key)
    if (value == []):
        wait_keys[key] = 2
    else:
        wait_keys[key] = np.inf
