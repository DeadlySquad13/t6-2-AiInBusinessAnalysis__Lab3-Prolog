% main.pl

:- initialization(main).

main :-
    load_module(db1),
    load_module(db2),
    load_module(db3),
    writeln('The system is ready to work. Use:'),
	writeln('scientific_ask(Query) - for scientific queries'),
	writeln('mythological_ask(Query) - for mythological queries'),
	writeln('religious_ask(Query) - for religious queries').

load_module(Module) :-
    atomic_list_concat([Module, '.pl'], FileName),
    (exists_file(FileName)
        -> consult(FileName),
           format('Module ~w loaded.~n', [Module])
        ;  format('FAIL: File ~w not found!~n', [FileName]),
           fail).

scientific_ask(Concept) :-
    findall(Description, db1:scientific_fact(Concept, Description), Results),
    print_solutions(Results).

mythological_ask(God) :-
    findall(Description, db2:mythological_fact(God, Description), Results),
    print_solutions(Results).

religious_ask(Religion) :-
    findall(Description, db3:religious_fact(Religion, Description), Results),
    print_solutions(Results).

print_solutions([]) :- writeln('No solution.').
print_solutions([H|T]) :-
    writeln(H),
    (T = [] -> true ; print_solutions(T)).
