# Context
This is a continuation of a series of hobby projects I have done throughout the years.
The overarching theme is artificial intelligence and computational neuroscience. 
Specific topics are amongst others: reinforcement learning, biologically plausible neural networks, simplified conceptual models of brains, decision making, language modelling, emergent behaviour and consciousness.
The focus is on understanding through building, and creating fascinating simulations and emergent 'life' as a hobby. Biological accuracy is not a hard requirement, and experiments may exist on various levels of abstraction.

# Current goal
I would love to have a brain model that I can tinker with to make increasingly complex, and optimize in a biologically plausible manner. This needs a rich adjustable environment that supports interesting features to interacting with different stages and complexities of the brain.

## Environment
The environment rules and complexity might change between experiments, to match how advanced or not the brain model is, or better fit a certain experiment.
I hope to enable large scales by keeping dimensionality of the world and senses low: No 3D environments with rigid-body physics and video input, like Minecraft AI agents
However, do think it would be really 'cool' and visually interesting to build up to a more advanced community with tools, collaboration and tasks requiring long-term planning: More interesting than microorganism-style simulations like Bibites.
My current middle ground I am thinking of is to build upon (either the source code or concepts of) one hour one life, as it by design models as much of the complexities of society, technology and human life that can be fit into a 2D game.

## Examples
Some experiments that I would like to do:
- (easy, achieved)Evolved in-out network: simulate a very simple neural network that is learned through evolution, environment not more complicated than hunger, food, pain, move fast or slow.
- (easy, achieved) Hebbian learning: same network as basis, but learnable connections to associate with colors, sounds etc.
- (easy? implementation dependent with how timing works) STDP instead of Hebbian to ensure sequences are learned in the right order of causation
- (moderately difficult, partially achieved) reinforcement learning and action selection with basal ganglia model
- (easy once previous done, compute cost difficult) Joint evolution of high-level network architecture and shape, limited "hardcoded" reflexes whilst reinforcement learning throughout life
- possibly cortex simulations
--- This is the gap that I hard to bridge, and likely requires scaling up and some manual tuning to create circumstances that foster emergence of:
- Visual cortex connectivity (i read some paper that it partially emerges from some learning rules)
- (hard, not personally achieved in biologically plausible manner) Empathy and altruism (e.g. not eating something when hungry fellow is nearby)
- (hard, not personally achieved) Cross-generational learning: learning that something is possible by seeing someone else do it
- (hard, not personally achieved) Evolution of communication signals (e.g. predator is coming, need assistance)
- (very hard, but the holy grail of what I have been wanting to see for years with a self-built brain model that I understand) further evolution from the above building blocks in e.g. one hour one life to the point of communities forming with somewhat advanced tech and communication that clearly outgrows evolution

# How?
- I don't want to reinvent the wheel, and (as long as it supports the above ideas, while handing me control, understanding, interesting visualisations, and imposing little to no constraints on my experiments) making use of frameworks would be perfect.
	- I wonder if Nengo is suitable for what I want, and not to clunky or ui-heavy (e.g. would it support parallel agents evolving without spending too much compute on individual spike simulations etc? both hardcoded and learned rules?)
	- I would love to use the richness of the One hour one life game, with its different types of plants, materials, recipes, technologies, temperature environments, hunger rules, aging etc. But am not sure whether it's engine can be reused (or rebuilt/translated?) for all the different environment settings I  would like to use. Perhaps only extracting its ruleset and artwork is an option? 
- I further need to determine how to encode the observation to the creature (possibly changing per experiment) on the balance of information richness and feasible dimensionality. 
Currently leaning towards making movement and observations relative to the direction that someone is facing. 2D world is rendered on the field of vision like 2.5D sprite based video games, with varying resolution that might be only 10 pixels wide initially, to prevent having to use large CNNs.

In general, I am fine with starting with heavily simplified environments and models, as long as it is possible to gradually increase the complexity and scale without switching to a different brain architecture orenvironment


# Learning resources for me
How to build a brain (spaun)
https://CompCogNeuro.org
https://forum.nengo.ai/
Neuromatch Academy