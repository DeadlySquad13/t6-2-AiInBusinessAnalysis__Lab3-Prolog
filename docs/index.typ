// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = [
  #line(start: (25%,0%), end: (75%,0%))
]

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): block.with(
    fill: luma(230), 
    width: 100%, 
    inset: 8pt, 
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.amount
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == "string" {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == "content" {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

#show figure: it => {
  if type(it.kind) != "string" {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    new_title_block +
    old_callout.body.children.at(1))
}

#show ref: it => locate(loc => {
  let target = query(it.target, loc).first()
  if it.at("supplement", default: none) == none {
    it
    return
  }

  let sup = it.supplement.text.matches(regex("^45127368-afa1-446a-820f-fc64c546b2c5%(.*)")).at(0, default: none)
  if sup != none {
    let parent_id = sup.captures.first()
    let parent_figure = query(label(parent_id), loc).first()
    let parent_location = parent_figure.location()

    let counters = numbering(
      parent_figure.at("numbering"), 
      ..parent_figure.at("counter").at(parent_location))
      
    let subcounter = numbering(
      target.at("numbering"),
      ..target.at("counter").at(target.location()))
    
    // NOTE there's a nonbreaking space in the block below
    link(target.location(), [#parent_figure.at("supplement") #counters#subcounter])
  } else {
    it
  }
})

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      block(
        inset: 1pt, 
        width: 100%, 
        block(fill: white, width: 100%, inset: 8pt, body)))
}



#let article(
  title: none,
  authors: none,
  date: none,
  abstract: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: (),
  fontsize: 11pt,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: "1",
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)

  if title != none {
    align(center)[#block(inset: 2em)[
      #text(weight: "bold", size: 1.5em)[#title]
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[Abstract] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}
#show: doc => article(
  font: ("Agbalumo",),
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)


#block[
МОСКОВСКИЙ ГОСУДАРСТВЕННЫЙ ТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ
им. Н.Э. Баумана
~
~
Факультет «Информатика и системы управления»
Кафедра «Систем обработки информации и управления»
~
~
~
~
ОТЧЁТ
~
Лабораторная работа №#strong[1]
по дисциплине «Разработка нейросетевых систем»
~
~
Тема: «Введение в DL»
Вариант 2
~
~
~
~
~
ИСПОЛНИТЕЛЬ:~ ~~~~~~~~~~~~~~\_Борисочкин М. И.\_\_
ФИО
группа ИУ5-21М~~~~~~~~~~~~~~~~ \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
подпись
~
\"#strong[\"];\_\_\_\_\_\_\_2024 г.
~
ПРЕПОДАВАТЕЛЬ:~~ ~~~~~~~~~~\_\_\_\_Канев А. И.\_\_\_\_
ФИО
~\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
подпись
~
\"#strong[\"];\_\_\_\_\_\_\_2024 г.
~
~
~
~
~
~
Москва~ -~ 2024
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
]
== Задание
<задание>
На основе рассмотренного на лекции примера реализуйте алгоритм Policy Iteration для любой среды обучения с подкреплением \(кроме рассмотренной на лекции среды Toy Text / Frozen Lake) из библиотеки Gym \(или аналогичной библиотеки).

== Выполнение
<выполнение>
Исходный код программы:

```python
from pprint import pprint

import gym
import numpy as np
from toy_environment.consts import CLIFF_WALKING_ENV

ENV = CLIFF_WALKING_ENV


class PolicyIterationAgent:
    """
    Класс, эмулирующий работу агента
    """

    def __init__(self, env):
        self.env = env
        # Пространство состояний
        self.observation_dim = ENV["observation_dim"]
        self.actions_variants = np.array(ENV["actions_variants"])
        # Задание стратегии (политики)
        # Карта 4х4 и 4 возможных действия
        self.policy_probs = np.full(
            (self.observation_dim, len(self.actions_variants)), 1 / len(ENV["actions_variants"])
        )
        # Начальные значения для v(s)
        self.state_values = np.zeros(shape=(self.observation_dim))
        # Начальные значения параметров
        self.maxNumberOfIterations = 100_000
        self.theta = 1e-6
        self.gamma = 0.99

    def print_policy(self):
        """
        Вывод матриц стратегии
        """
        print("Стратегия:")
        pprint(self.policy_probs)

    def policy_evaluation(self):
        """
        Оценивание стратегии
        """
        # Предыдущее значение функции ценности
        valueFunctionVector = self.state_values
        for iterations in range(self.maxNumberOfIterations):
            # Новое значение функции ценности
            valueFunctionVectorNextIteration = np.zeros(shape=(self.observation_dim))
            # Цикл по состояниям
            for state in range(self.observation_dim):
                # Вероятности действий
                action_probabilities = self.policy_probs[state]
                # Цикл по действиям
                outerSum = 0
                for action, prob in enumerate(action_probabilities):
                    innerSum = 0
                    # Цикл по вероятностям действий
                    for probability, next_state, reward, isTerminalState in self.env.P[
                        state
                    ][action]:
                        innerSum = innerSum + probability * (
                            reward + self.gamma * self.state_values[next_state]
                        )
                    outerSum = outerSum + self.policy_probs[state][action] * innerSum
                valueFunctionVectorNextIteration[state] = outerSum
            if (
                np.max(np.abs(valueFunctionVectorNextIteration - valueFunctionVector))
                < self.theta
            ):
                # Проверка сходимости алгоритма
                valueFunctionVector = valueFunctionVectorNextIteration
                break
            valueFunctionVector = valueFunctionVectorNextIteration
        return valueFunctionVector

    def policy_improvement(self):
        """
        Улучшение стратегии
        """
        qvaluesMatrix = np.zeros((self.observation_dim, len(self.actions_variants)))
        improvedPolicy = np.zeros((self.observation_dim, len(self.actions_variants)))
        # Цикл по состояниям
        for state in range(self.observation_dim):
            for action in range(len(self.actions_variants)):
                for probability, next_state, reward, isTerminalState in self.env.P[
                    state
                ][action]:
                    qvaluesMatrix[state, action] = qvaluesMatrix[
                        state, action
                    ] + probability * (
                        reward + self.gamma * self.state_values[next_state]
                    )

            # Находим лучшие индексы
            bestActionIndex = np.where(
                qvaluesMatrix[state, :] == np.max(qvaluesMatrix[state, :])
            )
            # Обновление стратегии
            improvedPolicy[state, bestActionIndex] = 1 / np.size(bestActionIndex)
        return improvedPolicy

    def policy_iteration(self, cnt):
        """
        Основная реализация алгоритма
        """
        policy_stable = False
        for i in range(1, cnt + 1):
            self.state_values = self.policy_evaluation()
            self.policy_probs = self.policy_improvement()
        print(f"Алгоритм выполнился за {i} шагов.")


def play_agent(agent):
    env = gym.make(ENV["name"], render_mode="human")
    state = env.reset()[0]
    done = False
    while not done:
        p = agent.policy_probs[state]
        if isinstance(p, np.ndarray):
            action = np.random.choice(len(agent.actions_variants), p=p)
        else:
            action = p
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        state = next_state
        if terminated or truncated:
            done = True


def main():
    # Создание среды
    env = gym.make(ENV["name"])
    env.reset()

    # Обучение агента
    agent = PolicyIterationAgent(env)
    agent.print_policy()
    agent.policy_iteration(100_000)
    agent.print_policy()

    # Проигрывание сцены для обученного агента
    play_agent(agent)


if __name__ == "__main__":
    main()
```

И содержимое `toy_environment.consts`:

```python
# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
FROZEN_LAKE_ENV = {
    "name": "FrozenLake-v1",
    "observation_dim": 16,
    # Массив действий в соответствии с документацией
    "actions_variants": [0, 1, 2, 3],
}

# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
CLIFF_WALKING_ENV = {
    "name": "CliffWalking-v0",
    "observation_dim": 48,
    # Массив действий в соответствии с документацией
    "actions_variants": [0, 1, 2, 3],
}
```
