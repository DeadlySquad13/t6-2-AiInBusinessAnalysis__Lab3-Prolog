% scientific module
:- module(db1, [ scientific_fact/2,
orbits/2,
planet/1,
moon/1,
star/1,
galaxy/1,
animal/1,
mammal/1,
vertebrate/1,
invertebrate/1,
habitat/2,
breathe/2,
element/1,
compound_item/1,
consists_of/2,
state/2,
force/1,
acts_on/2,
unit/1,
quantity/2,
contains/2,
orbital_body/1,
celestial_body/1,
inhabit/2,
living_being/1,
gas/1,
solid/1,
liquid/1,
exists_in/2,
measure/2
]).
:- discontiguous orbits/2.
:- discontiguous moon/1.
:- discontiguous force/1.
:- discontiguous acts_on/2.
:- discontiguous unit/1.
:- discontiguous quantity/2.
:- discontiguous animal/1.
:- discontiguous habitat/2.
:- discontiguous element/1.
:- discontiguous state/2.
:- discontiguous star/1.
% dynamic scientific_fact/2.
scientific_fact(gravity, 'Сила притяжения между двумя телами.').
scientific_fact(evolution, 'Процесс изменения живых организмов с течением времени.').
scientific_fact(relativity, 'Теория, описывающая гравитацию как искривление пространства-времени.').
scientific_fact(cell_theory, 'Все живые организмы состоят из клеток.').
scientific_fact(atomic_theory, 'Все вещества состоят из атомов.').
planet(mercury).
planet(venus).
planet(earth).
planet(mars).
planet(jupiter).
planet(saturn).
planet(uranus).
planet(neptune).
orbits(mercury, sun).
orbits(venus, sun).
orbits(earth, sun).
orbits(mars, sun).
orbits(jupiter, sun).
orbits(saturn, sun).
orbits(uranus, sun).
orbits(neptune, sun).
moon(moon).
orbits(moon, earth).
star(sun).
galaxy(milky_way).
contains(milky_way, sun).
animal(human).
animal(dog).
animal(cat).
animal(elephant).
animal(dolphin).
animal(fish).
animal(bird).
animal(snake).
mammal(human).
mammal(dog).
mammal(cat).
mammal(elephant).
mammal(dolphin).
vertebrate(human).
vertebrate(dog).
vertebrate(cat).
vertebrate(elephant).
vertebrate(dolphin).
invertebrate(octopus).
invertebrate(spider).
habitat(human, land).
habitat(dog, land).
habitat(cat, land).
habitat(elephant, land).
habitat(dolphin, water).
habitat(octopus, water).
habitat(alligator, water).
habitat(squirrel, land).
habitat(tiger, land).
habitat(monkey, land).
habitat(spider, land).
breathe(human, air).
breathe(dog, air).
breathe(cat, air).
breathe(elephant, air).
breathe(dolphin, air).
breathe(octopus, water).
breathe(spider, air).
element(hydrogen).
element(helium).
element(lithium).
element(beryllium).
element(boron).
element(carbon).
element(nitrogen).
element(oxygen).
element(fluorine).
element(neon).
element(sodium).
element(magnesium).
element(phosphorus).
element(sulfur).
element(chlorine).
compound_item(water).
compound_item(carbon_dioxide).
consists_of(water, hydrogen).
consists_of(water, oxygen).
consists_of(carbon_dioxide, carbon).
consists_of(carbon_dioxide, oxygen).
state(hydrogen, gas).
state(helium, gas).
state(lithium, solid).
state(beryllium, solid).
state(boron, solid).
state(carbon, solid).
state(nitrogen, gas).
state(oxygen, gas).
state(fluorine, gas).
state(neon, gas).
state(water, liquid).
state(carbon_dioxide, gas).
force(gravity).
force(electromagnetism).
force(strong_nuclear).
force(weak_nuclear).
acts_on(gravity, mass).
acts_on(electromagnetism, charge).
acts_on(strong_nuclear, quarks).
acts_on(weak_nuclear, subatomic_particles).
unit(meter).
unit(kilogram).
unit(second).
unit(ampere).
unit(kelvin).
unit(mole).
unit(candela).
unit(newton).
unit(pascal).
unit(joule).
quantity(length, meter).
quantity(mass, kilogram).
quantity(time, second).
quantity(electric_current, ampere).
quantity(temperature, kelvin).
quantity(amount_of_substance, mole).
quantity(luminous_intensity, candela).
quantity(force, newton).
quantity(pressure, pascal).
quantity(energy, joule).
orbital_body(X) :- planet(X).
orbital_body(X) :- moon(X).
celestial_body(X) :- star(X).
celestial_body(X) :- planet(X).
celestial_body(X) :- moon(X).
inhabit(X, Y) :- animal(X), habitat(X, Y).
living_being(X) :- animal(X).
gas(X) :- state(X, gas).
solid(X) :- state(X, solid).
liquid(X) :- state(X, liquid).
exists_in(X, Y) :- compound_item(X), consists_of(X, Y).
measure(X, Y) :- quantity(X, Y).
