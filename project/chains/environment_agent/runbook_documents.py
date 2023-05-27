from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

source1 = """
Let it grow.
More than 70 towns across the U.S. have signed on to relax maintenance rules for property owners during the month of May as part of a movement meant to feed local bee populations for the upcoming growing season.
Called “No Mow May,” the initiative started in the United Kingdom and was later adopted by Appleton, Wisconsin, which became the first U.S. town to implement it in 2020. This year, dozens of communities across the U.S. have adopted No Mow May to encourage citizens to let bee-friendly plants grow in their yards.
“Pesky weeds like clovers and dandelions are like cheeseburgers for bees,” said Israel Del Toro, an assistant biology professor at Lawrence University and city council member in Appleton.
And Appleton has seen the bee benefits, Del Toro said. Since the project began, No Mow May lawns in Appleton showed a fivefold increase in bee abundance and a threefold increase in bee diversity compared to nearby parkland that was mowed regularly.
Bees are essential pollinators. Every year, honeybees pollinate $15 billion worth of crops in the U.S. and help farmers produce about one-third of all food eaten by Americans.
But global bee populations are in danger — factors such as habitat loss, overuse of chemical pesticides and herbicides, disease and climate change can negatively impact bees, Del Toro said. Globally, 35% of invertebrate pollinators, mainly bees and butterflies, face extinction.
“If we can do just a little bit to help bees by not mowing our lawns and giving them extra food, that can go a long way,” Del Toro said.
Without regular lawn mowing, homeowners also reduce the risk of disturbing soil for bees that nest in the ground, such as mining or leafcutter bees, said Relena Ribbons, an assistant geosciences professor at Lawrence University.
Less mowing has the added benefit of reducing local air pollution. Since most lawn mowers are gas-powered, they can emit significant levels of air pollutants such as nitrous oxides, carbon monoxide, particulate matter and carbon dioxide.
Angela Vanden Elzen, a librarian and lifelong resident of Appleton, said she rarely saw bees around town growing up. Since No Mow May began, she has seen more bees than ever buzzing in Appleton’s lawns and gardens.
“No Mow May provides a way for us to help out our ecosystem with little effort,” Vanden Elzen said. “I love sharing the project with my kids. It started with honeybees and now they’ve grown more tolerant of all creatures — even spiders.”
Vanden Elzen’s 7-year-old daughter, Elora, said she loves participating in No Mow May with her community.
“I get excited about 'No Mow May' because we get to feed the bees,” Elora said. “And I like seeing all the bright colors in my front yard and seeing the bees buzzing on the flowers.”
In other towns, residents have expressed concerns about leaving their lawns unmowed. Jo Ann Litwin Clinton, the mayor of Orchard Park, New York, raised the issue of attracting ticks and rodents with unkempt lawns. Orchard Park citizens are currently debating whether mowing 2-by-2-foot sections is sufficient to encourage bee populations.
At the University of Minnesota, a group of researchers studying bees called the Bee Squad recommends practicing “Slow Mow Summer,” a method in which homeowners infrequently mow their lawns during summer, keeping grass moderately high. Elaine Evans, an extension professor at the University of Minnesota and researcher with the Bee Squad, said this strategy can help provide food for bees throughout the season — not just in May.
Matthew Normansell, an Appleton resident who runs Eden Wild Food, a foraging and wild food education business, said that No Mow May is an important part of the solution.
“It’s a first step in increasing awareness that our land and property is connected to the environment,” he said. “I don’t think it’s the answer, but it’s the beginning of an important conversation.”

"""

doc1 = Document(page_content=source1, metadata={"topic": "bees"})

source2 = """
The rapid development of unexpected drought, called flash drought, can severely impact agricultural and ecological systems with ripple effects that extend even further. Researchers at the University of Oklahoma are assessing how our warming climate will affect the frequency of flash droughts and the risk to croplands globally.

Jordan Christian, a postdoctoral researcher, is the lead author of the study, "Global projections of flash drought show increased risk in a warming climate," published today in Nature Communications Earth and Environment.

"In this study, projected changes in flash drought frequency and cropland risk from flash drought are quantified using global climate model simulations," Christian said. "We find that flash drought occurrence is expected to increase globally among all scenarios, with the sharpest increases seen in scenarios with higher radiative forcing and greater fossil fuel usage."

Radiative forcing describes the imbalance of radiation where more radiation enters Earth's atmosphere than leaves it. Like burning fossil fuels, these activities are among the most significant contributors to climate warming. The changing climate is expected to increase severe weather events from storms, flash flooding, flash droughts and more.

"Flash drought risk over cropland is expected to increase globally, with the largest increases projected across North America and Europe," Christian said.

"CMIP6 models projected a 1.5 times increase in the annual risk of flash droughts over croplands across North America by 2100, from the 2015 baseline of a 32% yearly risk in 2015 to 49% in 2100, while Europe is expected to have the largest increase in the most extreme emissions scenario (32% to 53%), a 1.7 times increase in annual risk," he said.

Jeffrey Basara, an associate professor in the School of Meteorology in the College of Atmospheric and Geographic Sciences and the School of Civil Engineering and Environmental Sciences in the Gallogly College of Engineering, is Christian's faculty advisor and study co-author. Basara is the executive associate director of the hydrology and water security program and leads OU's Climate, Hydrology, Ecosystems and Weather research group. The researchers have been investigating ways to improve flash drought identification and prediction since 2017, with multiple papers published in the Journal of Hydrometeorology, Environmental Research Letters and Nature Communications.

"This study continues to emphasize that agricultural producers, both domestic and abroad, will face increasing risks associated with water availability due to the rapid development of drought. As a result, socioeconomic pressures associated with food production, including higher prices and social unrest, will also increase when crop losses occur due to flash drought," Basara said.
"""

doc2 = Document(page_content=source2, metadata={"topic": "drought"})

source3 = """
The future of Samoa's electricity system could go green, a University of Otago study has shown.

Pacific Island nations are particularly susceptible to climate change and face high costs and energy security issues from imported fossil fuels.

For these reasons many Pacific Island nations have developed ambitious 100 per cent renewable energy targets. However, they have not been subject to rigorous peer-reviewed studies to help develop these targets and pathways for achieving them in the same way as more developed countries.

To meet this need, Otago Energy Science and Technology Masters student Tupuivao Vaiaso mapped future scenarios for Samoa's electricity system by carefully balancing renewable supply and electricity demand.

The study, published in Renewable and Sustainable Energy Reviews, shows high proportions (above 90 per cent) of renewable generation coupled with battery or pumped hydro energy storage is technically feasible, and economically viable -- it would cost far less than current electricity production.

A special combination of characteristics makes this possible: great solar resources with very little seasonal variation; little variation in demand due to constant seasonal conditions; and high electricity prices from the current use of diesel.

However, the results also show there is a significant trade-off between percentage of renewable supply and affordability. To counter this, Mr Vaiaso says targets should be set just below 100 per cent for it to be economically attractive and/or biomass options (like the recent biomass gasification plant) could potentially be considered for the last few percent.

Co-author Associate Professor Michael Jack, Director of the Energy Management programme in the Department of Physics, says the study shows an affordable pathway to achieve renewable energy targets which is important as cutting emissions must be achieved in a way that does not increase energy hardship.

"These results have important implications for energy policy directions for Samoa and are directly applicable to many other countries in the Pacific," he says.

He believes the study also suggests Pacific Island nations are an important option for the growing trend in international sustainable finance.

"The high price of electricity means that renewable options are much more economically attractive than many other countries. These are essentially low hanging fruit that provide multiple benefits and, I imagine, would be attractive to these funding bodies."

"""

doc3 = Document(page_content=source3, metadata={"topic": "renewables"})


########################### TEMPLATE BELOW

source0 = """How to Handle a Disk Error:

Disk errors mean the entire VM has reached a bad state. We need to reboot the entire VM by running this command: 

"""

doc0 = Document(page_content=source0, metadata={"topic": "environment"})

########################## TEMPLATE ABOVE

DOCS = [doc1, doc2, doc3]

def create_index():
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in DOCS:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings())
    return search_index