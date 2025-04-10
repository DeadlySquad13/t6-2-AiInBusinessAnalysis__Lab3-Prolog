% religious module

:- module(db3, [ religious_fact/2,
                  rel_deity/1,
                  apostle/1,
                  angel/1,
                  prophet/1,
                  human/1,
                  evil_being/1,
                  scripture/1,
                  rel_contains/2,
                  part_of/2,
                  worship_place/1,
                  religion/1,
                  rel_associated_with/2,
                  located_in_religion/2,
                  ritual/1,
                  holiday/1,
                  celebrates/2,
                  holy/1,
                  text/1,
                  worship_site/1,
                  celebration/1,
                  related_to/2,
                  part/2,
                  participant_in/2,
                  sacred_text_part/2,
                  evil/1,
                  human_believer/1
                ]).

:- discontiguous rel_deity/1.
:- discontiguous apostle/1.
:- discontiguous angel/1.
:- discontiguous prophet/1.
:- discontiguous human/1.
:- discontiguous evil_being/1.
:- discontiguous scripture/1.
:- discontiguous rel_contains/2.
:- discontiguous part_of/2.
:- discontiguous worship_place/1.
:- discontiguous religion/1.
:- discontiguous rel_associated_with/2.
:- discontiguous located_in_religion/2.
:- discontiguous ritual/1.
:- discontiguous holiday/1.
:- discontiguous celebrates/2.

religious_fact(christianity, 'Религия, основанная на учении Иисуса Христа.').
religious_fact(islam, 'Религия, основанная на учении пророка Мухаммеда.').
religious_fact(hinduism, 'Одна из старейших религий, основанная на ведах.').
religious_fact(buddhism, 'Религия, основанная на учениях Будды.').
religious_fact(judaism, 'Одна из древнейших монотеистических религий.').

rel_deity(god).
rel_deity(allah).
rel_deity(brahma).
rel_deity(vishnu).
rel_deity(shiva).
rel_deity(jesus).
rel_deity(rama).
rel_deity(zeus).
rel_deity(thor).

apostle(peter).
apostle(paul).

angel(gabriel).
angel(michael).
angel(raphael).

prophet(moses).
prophet(isaiah).
prophet(muhammad).
prophet(krishna).
prophet(buddha).
prophet(confucius).
prophet(zarathustra).

human(adam).
human(eve).
human(abraham).
human(sarah).
human(noah).
human(mary).

evil_being(satan).
evil_being(lucifer).
evil_being(demon).
evil_being(ahriman).

scripture(bible).
scripture(quran).
scripture(vedas).
scripture(book_of_mormon).
scripture(tripitaka).
scripture(avesta).

rel_contains(bible, old_testament).
rel_contains(bible, new_testament).
rel_contains(quran, surah).
rel_contains(book_of_mormon, book_of_mormon_parts).
rel_contains(tripitaka, sutras).
rel_contains(avesta, gathas).

part_of(old_testament, bible).
part_of(new_testament, bible).
part_of(surah, quran).
part_of(book_of_mormon_parts, book_of_mormon).
part_of(sutras, tripitaka).
part_of(gathas, avesta).

worship_place(church).
worship_place(mosque).
worship_place(temple).
worship_place(synagogue).
worship_place(gurdwara).
worship_place(pagoda).

religion(christianity).
religion(islam).
religion(hinduism).
religion(judaism).
religion(sikhism).
religion(buddhism).
religion(zoroastrianism).

rel_associated_with(church, christianity).
rel_associated_with(mosque, islam).
rel_associated_with(temple, hinduism).
rel_associated_with(synagogue, judaism).
rel_associated_with(gurdwara, sikhism).
rel_associated_with(pagoda, buddhism).

located_in_religion(vatican, church).
located_in_religion(mecca, mosque).
located_in_religion(varanasi, temple).
located_in_religion(jerusalem, synagogue).
located_in_religion(amritsar, gurdwara).
located_in_religion(nara, pagoda).

ritual(prayer).
ritual(fasting).
ritual(pilgrimage).
ritual(sacrifice).
ritual(circumcision).
ritual(langar).
ritual(zen_meditation).
ritual(yasna).

holiday(christmas).
holiday(easter).
holiday(ramadan).
holiday(diwali).
holiday(yom_kippur).
holiday(vaisakhi).
holiday(vesak).
holiday(nawruz).
holiday(shabbat).

celebrates(christmas, birth_of_jesus).
celebrates(easter, resurrection_of_jesus).
celebrates(ramadan, revelation_of_quran).
celebrates(diwali, return_of_rama).
celebrates(yom_kippur, atonement).
celebrates(vaisakhi, birth_of_khalsa).
celebrates(vesak, buddha_day).
celebrates(nawruz, persian_new_year).
celebrates(shabbat, seventh_day_in_judaism).

holy(X) :- rel_deity(X).
holy(X) :- angel(X).
holy(X) :- prophet(X).

text(X) :- scripture(X).
worship_site(X) :- worship_place(X).
celebration(X) :- holiday(X).
related_to(X, Y) :- rel_associated_with(X, Y).
part(X, Y) :- rel_contains(Y, X).
participant_in(X, Y) :- ritual(Y), religion(X).
sacred_text_part(X, Y) :- part_of(X, Y).
evil(X) :- evil_being(X).
human_believer(X) :- human(X).
