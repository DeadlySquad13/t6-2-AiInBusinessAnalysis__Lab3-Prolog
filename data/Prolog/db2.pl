% mythological module

:- module(db2, [ mythological_fact/2,
                  deity/1,
                  pantheon/1,
                  belongs_to/2,
                  creature/1,
                  married/2,
                  loves/2,
                  parent/2,
                  origin/2,
                  hero/1,
                  associated_with/2,
                  place/1,
                  located_in/2,
                  mythical_being/1,
                  belongs/2,
                  mythical_place/1,
                  part_of_pantheon/2,
                  inhabits/2,
                  hero_of/2
                ]).

:- discontiguous deity/1.
:- discontiguous married/2.
:- discontiguous loves/2.
:- discontiguous parent/2.
:- discontiguous origin/2.
:- discontiguous associated_with/2.
:- discontiguous located_in/2.
:- discontiguous creature/1.

mythological_fact(zeus, 'Греческий бог грома и молний.').
mythological_fact(thor, 'Скандинавский бог грома.').
mythological_fact(odin, 'Главный бог в скандинавской мифологии.').
mythological_fact(ra, 'Древнеегипетский бог солнца.').
mythological_fact(poseidon, 'Греческий бог моря.').
mythological_fact(frey, 'Богиня любви, жительница Асгарда в скандинавской мифологии.').

deity(zeus).
deity(hera).
deity(poseidon).
deity(hades).
deity(athena).
deity(apollo).
deity(aphrodite).
deity(ares).
deity(demeter).
deity(artemis).
deity(hecate).
deity(odin).
deity(frey).
deity(frigg).
deity(thor).
deity(loki).
deity(freya).
deity(tyr).
deity(heimdall).
deity(baldr).

pantheon(greek).
pantheon(norse).

belongs_to(zeus, greek).
belongs_to(hera, greek).
belongs_to(poseidon, greek).
belongs_to(hades, greek).
belongs_to(athena, greek).
belongs_to(apollo, greek).
belongs_to(aphrodite, greek).
belongs_to(ares, greek).
belongs_to(demeter, greek).
belongs_to(artemis, greek).
belongs_to(hecate, greek).
belongs_to(odin, norse).
belongs_to(frey, norse).
belongs_to(frigg, norse).
belongs_to(thor, norse).
belongs_to(loki, norse).
belongs_to(freya, norse).
belongs_to(tyr, norse).
belongs_to(heimdall, norse).
belongs_to(baldr, norse).

creature(centaur).
creature(minotaur).
creature(cerberus).
creature(hydra).
creature(dragon).
creature(fenrir).
creature(jormungandr).
creature(phoenix).
creature(sphinx).
creature(griffin).
creature(pegasus).
creature(cyclops).
creature(siren).
creature(kraken).
creature(yeti).

married(aphrodite, hephaestus).

loves(aphrodite, ares).
loves(aphrodite, adonis).

parent(zeus, ares).
parent(hera, ares).

origin(centaur, greek).
origin(minotaur, greek).
origin(cerberus, greek).
origin(hydra, greek).
origin(dragon, various).
origin(fenrir, norse).
origin(jormungandr, norse).
origin(phoenix, egyptian).
origin(sphinx, egyptian).

hero(hercules).
hero(theseus).
hero(achilles).
hero(odysseus).
hero(sigurd).
hero(beowulf).

associated_with(hercules, greek).
associated_with(theseus, greek).
associated_with(achilles, greek).
associated_with(odysseus, greek).
associated_with(sigurd, norse).
associated_with(beowulf, norse).

place(olympus).
place(asgard).
place(midgard).
place(tartarus).
place(valhalla).
place(elisium).

located_in(olympus, greek).
located_in(asgard, norse).
located_in(midgard, norse).
located_in(tartarus, greek).
located_in(valhalla, norse).
located_in(elisium, greek).

mythical_being(X) :- deity(X).
mythical_being(X) :- creature(X).
mythical_being(X) :- hero(X).

belongs(X, Y) :- deity(X), pantheon(Y), belongs_to(X, Y).
belongs(X, Y) :- hero(X), associated_with(X, Y).
belongs(X, Y) :- creature(X), origin(X, Y).

mythical_place(X) :- place(X).

part_of_pantheon(X, Y) :- deity(X), belongs_to(X, Y).

inhabits(X, Y) :-
    deity(X),
    place(Y),
    located_in(Y, Z),
    belongs_to(X, Z).
inhabits(X, Y) :-
    creature(X),
    place(Y),
    located_in(Y, Z),
    origin(X, Z).

hero_of(X, Y) :- hero(X), associated_with(X, Y).
