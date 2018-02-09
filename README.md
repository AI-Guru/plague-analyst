# plague-analyst

Deep Learning and diseases! Non-deadly ones...

This is the git-repo for my blogpost. Read it if you want to know more!
[Read the blog.](http://ai-guru.de/index.php/deep-learning-and-diseases/)

## Synopsis

The plague-analyst shows two things:
- How to simulate epidemics based on the [SIR-model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology).
- How to train Neural Networks that predict properties of diseases based on SIR-data.

SIR stands for: **Susceptible** are people that can be infected, **Infected** are people that are ill and spread the disease, and **Recovered** are people that are currently immune.

Diseases can be simulated with cellular automata. This project provides functionality to simulate the progression of diseases in order to create data-sets for Neural Network training.

This is a picture of such a cellular automaton over time. It shows how a disease spreads in a population:
![alt Cellular automaton.](https://github.com/AI-Guru/plague-analyst/blob/master/readme/output-1.gif)

And this diagram shows the progression as a function:
![alt SIR diagram.](https://github.com/AI-Guru/plague-analyst/blob/master/readme/output-1.png)

## Code Example

This code example shows how to run a single disease-simulation and render the outcome:

    import sir_dataset

    sir_simulator = SIRSimulator(50, 50, 100) # width, height, timesteps

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


## Running

To run the Neural Network training, just:

    python neuralnetwork.py

This is how a dense network trains:
![alt Dense.](https://github.com/AI-Guru/plague-analyst/blob/master/readme/plague-analyst-dense.jpg)

This is how a deep-GRU network trains:
![alt Deep GRU.](https://github.com/AI-Guru/plague-analyst/blob/master/readme/plague-analyst-deepgru.jpg)
