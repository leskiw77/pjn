import math
import pickle
import re
import sys
from os import listdir
from os.path import isdir, join

import numpy as np


class Word:
    def __init__(self, idx, line):
        self.idx = idx
        tmp = line.lower().replace('\n', '').split(';')

        self.representation, self.number = tmp[0], tmp[1]

        word_forms = tmp[2].split(':')
        word_base = word_forms[0]
        self.full_word_forms = set()
        self.full_word_forms.add(self.representation)
        for form in word_forms[1:]:
            if len(form) > 0 and form != "*":
                self.full_word_forms.add(word_base + form)

    def __eq__(self, other):
        return other.idx == self.idx

    def __repr__(self):
        # return '{} {} {}'.format(self.representation, self.idx, self.number)
        return '{} '.format(self.representation)

    def full_repr(self):
        return '{} {} {} {}'.format(self.representation, self.idx, self.number, self.full_word_forms)

    def __hash__(self):
        return self.idx


class SpeechClass:
    def __init__(self, file_name):
        self.id_to_word = dict()
        self.word_to_id = dict()

        with open(file_name) as f:
            for idx, line in enumerate(f.readlines()):
                w = Word(idx, line)
                self.id_to_word[idx] = w

                for word in w.full_word_forms:
                    self.word_to_id[word] = idx

    def has(self, word: str):
        # possible time complexity reduction
        return word.strip().lower() in self.word_to_id.keys()

    def get(self, word: str):
        return self.id_to_word[self.word_to_id[word.strip().lower()]]

    def get_all(self):
        return self.id_to_word.values()


nouns = SpeechClass(file_name='class_a.txt')
adjectives = SpeechClass(file_name='class_c.txt')

window = 4


def yield_adjective_noun(file_to_read):
    with open(file_to_read) as f:
        text = f.read()

    clean_pattern = re.compile('[1-9*!@#$,.():\-?";]')
    clean_array = [x for x in re.sub(clean_pattern, '', text).split() if len(x) > 3]
    array_len = len(clean_array)

    for i, word in enumerate(clean_array):
        if adjectives.has(word):

            for j in range(1, window):
                if i + j >= array_len:
                    break
                noun_expected = clean_array[i + j]

                if nouns.has(noun_expected):
                    adjective_object = adjectives.get(word)
                    noun_object = nouns.get(noun_expected)
                    yield adjective_object, noun_object


def process_document(file_to_read, adjectives_occurrence, nouns_occurrence):
    print("Parsing document: {}".format(file_to_read))
    for (a, n) in yield_adjective_noun(file_to_read):
        d1 = adjectives_occurrence.get(a)
        d1[n] = d1.get(n, 0) + 1

        d2 = nouns_occurrence.get(n)
        d2[a] = d2.get(a, 0) + 1


def normalize_vector(vec):
    normalization_factor = sum(list([v ** 2 for v in vec.values()]))
    normalization_factor = math.sqrt(normalization_factor)
    for k in vec.keys():
        vec[k] = vec[k] / normalization_factor
    return vec


def vector_for_noun(noun, nouns_occurrence, adjectives_occurrence):
    f_w = nouns.get(noun)

    word_vec = dict()
    for adj in nouns_occurrence[f_w]:
        word_vec[adj] = nouns_occurrence[f_w][adj] * adjectives_occurrence[adj][f_w]
    return normalize_vector(word_vec)


def vector_for_category(category, nouns_occurrence, adjectives_occurrence):
    dict_list = []
    for f in category:
        dict_list.append(vector_for_noun(f.lower(), nouns_occurrence, adjectives_occurrence))

    sums = {}
    for _dics in dict_list:
        for key, val in _dics.items():
            sums[key] = sums.get(key, 0) + val

    return normalize_vector(sums)


def vectors_distance(vec1, vec2):
    markers = {x: c for c, x in enumerate({**vec1, **vec2}.keys())}
    v1 = np.zeros(len(markers))
    v2 = np.zeros(len(markers))

    for word, val in vec1.items():
        index = markers[word]
        v1[index] = val

    for word, val in vec2.items():
        index = markers[word]
        v2[index] = val

    return np.linalg.norm(v1 - v2)


def read_data():
    mypath = "data"

    only_directories = [d for d in listdir(mypath) if isdir(join(mypath, d))]
    files_to_read = []

    for author in only_directories:
        for book in listdir(join(mypath, author)):
            files_to_read.append(join(mypath, author, book))

    files_to_read.sort()

    adjectives_occurrence = dict()
    for a in adjectives.get_all():
        adjectives_occurrence[a] = dict()

    nouns_occurrence = dict()
    for n in nouns.get_all():
        nouns_occurrence[n] = dict()

    for file_to_read in files_to_read:
        process_document(file_to_read, adjectives_occurrence, nouns_occurrence)

    for a in adjectives_occurrence:
        normalize_vector(adjectives_occurrence[a])

    for n in nouns_occurrence:
        normalize_vector(nouns_occurrence[n])

    with open("dump_nouns.bin", mode='wb') as binary_file:
        pickle.dump(nouns_occurrence, binary_file)

    with open("dump_adjectives.bin", mode='wb') as binary_file:
        pickle.dump(adjectives_occurrence, binary_file)

    return nouns_occurrence, adjectives_occurrence


def read_dump():
    print("Reading dump - start")
    with open("dump_nouns.bin", "rb") as f:
        nouns_occurrence = pickle.load(f)

    with open("dump_adjectives.bin", "rb") as f:
        adjectives_occurrence = pickle.load(f)
    print("Reading dump - completed")

    return nouns_occurrence, adjectives_occurrence


class Category:
    def __init__(self, name, words):
        self.words = words
        self.name = name

        self.vector = None

    def initialize_vector(self, nouns_occurrence, adjectives_occurrence):
        self.vector = vector_for_category(self.words, nouns_occurrence, adjectives_occurrence)

    def __eq__(self, other):
        return other.name == self.name

    def __repr__(self):
        return self.name


location = Category("miejsce", {'bank', 'budynek', 'dom', 'granica', 'klub', 'kraj', 'miasto', 'miejsce', 'ośrodek',
                                'pokój', 'szkoła', 'teatr', 'teren', 'ulica', 'rynek', 'most', "klasztor", "kasyno"})

substance = Category("substancja", {'asfalt', 'azot', 'ciecz', 'drewno', 'gaz', 'kryształ', 'marmur', 'materiał',
                                    'metal', 'minerał', 'płyn', 'substancja', 'tlen', 'tworzywo'})

action = Category("akcja", {"chodzenie", "bieganie", "skakanie", "zmywanie", "mycie", "szycie", "aktywność", "czynność",
                            "działanie", "akcja"})

animal = Category("zwierzęta", {"kot", "pies", "kura", "kurczak", "świnia", "ptak", "ssak", "gad", "owad", "wąż",
                                "komar", "szczur", "mysz"})

artefact = Category("przedmiot", {"przedmiot", "artefakt", "korona", "krzesło", "stół", "łóżko", "łyżka", "talerz",
                                  "widelec", "szczotka", "kubek", "zeszyt", "książka", "komputer"})

attribute = Category("akrybut", {"siła", "czułość", "pycha", "duma", "cecha", "dobroć"})

body = Category("ciało", {"głowa", "ręka", "noga", "ząb", "ciało", "palec", "włosy", "korpus", "tułów"})

cognition = Category("wiedza", {"wiedza", "mądrość", "znajomość", "poznanie", "umiejętność"})

communication = Category("komunikacja", {"komunikacja", "rozmowa", "mowa", "dyskusja", "kwestia", "porozumienie",
                                         "informacja"})

event = Category("wydarzenie", {"urodziny", "wesele", "impreza", "msza", "festyn", "chrzest", "zabawa"})

feeling = Category("uczucie", {"miłość", "ból", "pieczenie", "zadowolenie", "Rozbawienie", "Wesołość",
                               "Śmiałość", "Zainspirowanie", "Szczęście", "Wyciszenie", "Relaks", 'Akceptacja',
                               'Życzliwość', 'Zaspokojenie', 'Satysfakcja', 'Relaks', 'Wypoczęcie', 'Zainteresowanie',
                               'chęć', 'emocja', 'lęk', 'nadzieja', 'nienawiść', 'pewność', 'piękno',
                               'podziw', 'przyjaźń', 'radość', 'rezygnacja', 'strach', 'zło', 'złość'
                               })

food = Category("jedzenie",
                {'jabłko', 'gruszka', 'kapusta', 'czosnek', 'masło', 'chleb', 'jajko', 'sernik', 'ryż', 'makaron',
                 'pierogi', 'kaczka', 'obiad', 'danie', 'sól'})

group = Category("grupa", {"grupa", "kolekcja", "zestaw", "tłum", "zbiór", "zbiorowisko", "zespół"})

motive = Category('przyczyna', {'przyczyna', 'motyw', 'powód'})

natural_object = Category("obiekt naturalny",
                          {'las', 'góry', 'niebo', 'gwiazdy', 'wodospad', 'drzewo', 'kwiat', 'laka'})

natural_phenomenon = Category("zjawisko naturalne", {'pożar', 'powódź', 'katastrofa', 'huragan', 'tornado', 'lawina',
                                                     'zamieć', 'susza'})

person = Category("osoba", {"osoba", 'człowiek', "kobieta", "dziecko", "lekarz", "piekarz", "alergolog", "naukowiec",
                            "anglistyka"})

plant = Category("roślina", {"kwiatek", 'gałąź', 'pień', 'jabłoń', 'brzoza', 'tulipan', 'trawa', 'drzewo', 'roślina',
                             'tymianek', 'żywopłot'})

possession = Category("posiadanie", {'posiadanie', 'własność', 'dom', 'posesja', })

process = Category("process", {"proces", 'czynność', 'ładowanie', 'oczekiwanie', 'bluźnienie'})

quantity = Category("ilość", {'ilość', 'miara', 'waga', 'kilogram', 'litr', 'metr', })

relation = Category("relacja", {'miłość', 'flirt', 'kłótnia', 'zaręczyny', 'altruizm', 'bliskość', 'braterstwo'})

shape = Category("kształt", {'kwadrat', 'owal', 'trójkąt', 'prostokąt', 'stożek'})

state = Category("stan", {'stan', 'położenie'})

time = Category("czas", {"czas", 'minuta', 'godzina', 'sekunda', 'ciągłość', 'prędkość'})

# nouns_occurrence, adjectives_occurrence = read_data()
nouns_occurrence, adjectives_occurrence = read_dump()


categories = [location,
              substance,
              action,
              animal,
              artefact,
              attribute,
              body,
              cognition,
              communication,
              event,
              feeling,
              food,
              group,
              motive,
              natural_object,
              natural_phenomenon,
              person,
              plant,
              possession,
              process,
              quantity,
              relation,
              shape,
              state,
              time]

read_categories_dump = True

if read_categories_dump:
    with open("categories.bin", "rb") as f:
        categories = pickle.load(f)
else:
    iterations = 0

    if False:
        with open("categories.bin", "rb") as f:
            categories = pickle.load(f)

    [cat.initialize_vector(nouns_occurrence, adjectives_occurrence) for cat in categories]

    vec_coef = 0.9

    for i in range(iterations):
        length = len(nouns.get_all())
        j = 0
        for w in list(nouns.get_all()):
            j += 1
            if j % 1000 == 0:
                print('Iteration {}. Processed {} of {}'.format(i, j, length))

            v = vector_for_noun(w.representation, nouns_occurrence, adjectives_occurrence)
            distance_list = [(cat, vectors_distance(v, cat.vector)) for cat in categories]
            distance_list.sort(key=lambda x: x[1])

            if distance_list[0][1] / distance_list[1][1] < vec_coef:
                print(w.representation, distance_list[:3], distance_list[0][1] / distance_list[1][1])
                chosen_category = distance_list[0][0]
                chosen_category.words.add(w.representation)

        [cat.initialize_vector(nouns_occurrence, adjectives_occurrence) for cat in categories]

        with open("categories-{}-{}.txt".format(vec_coef, i), mode='w') as file:
            for c in categories:
                g = '{"' + "\", \"".join(c.words) + '"}'
                line = 'Category("{}", {})\n'.format(c.name, g)
                file.write(line)

        with open("categories-{}-{}.bin".format(vec_coef, i), mode='wb') as binary_file:
            pickle.dump(categories, binary_file)

    with open("categories.bin", mode='wb') as binary_file:
        pickle.dump(categories, binary_file)


while True:
    print("Enter keyword: ")
    line = sys.stdin.readline()
    word = line.lower().split()[0]
    print("Calculating distance for word: {}".format(word))

    try:
        v = vector_for_noun(word, nouns_occurrence, adjectives_occurrence)
        distance_list = [(cat, vectors_distance(v, cat.vector)) for cat in categories]
        distance_list.sort(key=lambda x: x[1])
        print(distance_list)
    except KeyError:
        print('No word {} in nouns dictionary')
