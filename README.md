# plague-analyst

Deep Learning and diseases! Non-deadly ones...

## Synopsis

The plague-analyst shows two things:
- Simulating epidemics based on the [SIR-model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology).
- Training Neural Networks that predict properties of diseases based on SIR-data.

SIR stands for: Susceptible are people that can be infected, Infected are people that are ill, and Recovered are people that are currently immune. Diseases can be simulated with cellular automata.

This project provides functionality to simulate the progression of diseases in order to create data-sets for Neural Network training.

This is a picture of such an automaton over time. It shows how a disease spreads in a population:
![alt text](https://github.com/AI-Guru/plague-analyst/blob/master/readme/output-1.gif)
The green

And this diagram shows the progression as a function:
![alt text](https://github.com/AI-Guru/plague-analyst/blob/master/readme/output-1.png)

## Code Example

This code example shows how to run a single disease-simulation and render the outcome:
    import sir_dataset

    sir_simulator = SIRSimulator(50, 50, 100)
    initially_infected = 5
    time_infection = 2
    time_recover = 4
    transmission_probability = 0.4
    average_contacts = 2

    states = sir_simulator.simulate(initially_infected, time_infection, time_recover, transmission_probability, average_contacts)
    render_states(states, "output.gif")
    counts = states_to_counts(states)
    render_counts("output.png", counts)


And this code example shows how to generate a whole data-set with 10000 entries:
    import sir_dataset

    sir_simulator = SIRSimulator(20, 20, 50)

    size = 10000
    dataset_type = "counts"

    dataset = sir_simulator.generate_dataset(
        size = size,
        split_ratio = "7:1:2",
        dataset_type = dataset_type,
        initially_infected = [5, 10, 20, 40],
        time_infection = [1, 2, 4, 8],
        time_recover = [2, 4, 8, 16],
        transmission_probability = [0.1, 0.2, 0.4, 0.8],
        average_contacts = [2, 4, 8, 16]
    )

    print_dataset_statistics(dataset)

    dataset_path = "dataset-{}-{}.p".format(dataset_type, size)
    save_dataset(dataset, dataset_path)


Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Motivation

A short description of the motivation behind the creation and maintenance of the project. This should explain **why** the project exists.

## Installation

Provide code examples and explanations of how to get the project.

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

## License

A short snippet describing the license (MIT, Apache, etc.)
