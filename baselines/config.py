# config.py

INPUT_FILES = [
    "Handbook_of_Unmanned_Aerial_Vehicles.mmd",
    "Handbook_of_Unmanned_Aerial_Vehicles_chapter6.mmd",
]

TOP_K = 200
OUTDIR = "data/test_run"  # None = save next to each input file

STOPWORDS = {
    'a','an','the','and','or','but','if','while','is','are','was','were','be','been','being',
    'to','of','in','on','for','with','by','as','at','from','into','through','during','including',
    'until','against','among','throughout','despite','towards','upon','about','above','below','up',
    'down','out','off','over','under','again','further','then','once','here','there','all','any',
    'both','each','few','more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','can','will','just','should','now','this','that','these','those','it',
    'its','their','his','her','our','your','you','we','they','he','she','them','us','i','me','my',
    'mine','ours','yours','also','may','might','must','would','could','shall','do','does','did',
    'have','has','had'
}