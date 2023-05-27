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

source4 = """
At Antelope Island State Park near Salt Lake City in the fall of 2022, three duck hunters dragged a sled across cracked desert sand in search of the water’s edge. The birds they sought were bunched in meager puddles far in the distance. Just to the west, the docks of an abandoned marina caved into the dust and a lone sailboat sat beached amid sagebrush.

“Biologists are worried that we’re on the brink of ecological collapse of the lake,” says Chad Yamane, the regional director of Ducks Unlimited, a nonprofit that conserves, restores and manages habitats for North America’s waterfowl, and a waterfowl hunter himself.

Last fall, the Great Salt Lake hit its lowest level since record keeping began. The lake’s elevation sank to nearly six meters below the long-term average, shriveling the Western Hemisphere’s largest saline lake to half its historic surface area. The lake’s shrinking threatens to upend the ecosystem, disrupting the migration and survival of 10 million birds, including ducks and geese.

Duck hunters aren’t the only ones worried about the Great Salt Lake. The decades-long decline in lake level is raising alarm bells for millions of people who live in the region. As the lake recedes, its namesake city and surrounding communities face a host of potential problems. The low lake level and increasing salinity threaten to disrupt economic mainstays like agriculture, tourism, mineral extraction and brine shrimp harvesting. Exposed sediments can also reduce air quality and so threaten public health. “It concerns everyone,” Yamane says. “It’s now on the forefront of every Utahan’s mind.”

And the Great Salt Lake isn’t unique. Many of the world’s saline lakes are facing a double whammy: People are taking more water from the tributaries that feed the lakes, while a hotter, drier climate means it takes longer to refill them.

Found on every continent, saline lakes include the Caspian Sea, the largest lake in the world, as well as the lowest, the Dead Sea. Saline lakes are terminal lakes — they have no drain, meaning no rivers flow out of them. As water evaporates, salts are left behind from the minerals that wash off the surrounding landscape. Since they are usually found in arid landscapes that receive little precipitation, saline lakes are the first in line to be affected by long-term droughts, which are becoming more common with climate change.

At the same time, the people who live in these deserts divert freshwater for crops, homes and industry. Residents siphon water from streams and rivers into canals, pipelines or reservoirs before it reaches the lakes. And as the lakes shrink, their concentration of salt increases.

Lake Poopó, a high-elevation lake in Bolivia that used to stretch 90 kilometers long and 32 kilometers wide, is now a salty mud flat. The Aral Sea in Kazakhstan and Uzbekistan, once the world’s fourth largest lake, has at times in recent decades withered to a tenth of its historic 68,000-square-kilometer surface area. India’s largest inland salt lake, Sambhar Salt Lake, is in severe decline, as is Africa’s Lake Chad. Some saline lakes, like Nevada’s Winnemucca Lake, dried up so long ago — the waters that fed it diverted to agricultural fields — that most people have forgotten they were ever wet.

According to a report released by researchers at Brigham Young University in January, the Great Salt Lake will likely also disappear within five years if residents continue their current rate of “unsustainable” water consumption.

The good news is Utahans still have time to halt or even reverse the Great Salt Lake’s decline by using less water. Cutting agricultural and other outdoor water use by a third to half through a combination of voluntary conservation measures and policy changes would allow the lake to refill enough to support the region’s economy, ecology and quality of life, the report says. If Utahans succeed, the Great Salt Lake can be a model for how to save other saline lakes around the world.

Why is the Great Salt Lake declining?

Like other terminal lakes, the Great Salt Lake naturally rises or falls based on how much water falls into and evaporates out of its watershed each year. Most of the precipitation falls as snow in the winter, melting each spring to fill streams that eventually empty into the lake. Because it’s surprisingly shallow for its size — an average of just over four meters deep — the Great Salt Lake fills or drains quickly. In the 1980s, for instance, a wet spell swelled the lake’s surface area to nearly 6,000 square kilometers, more than twice as big as it is today.

Despite natural variations, the lake level is undisputedly trending downward. Hydrologic modeling by Wayne Wurtsbaugh and Sarah Null, both experts on the lake based at Utah State University in Logan, shows that if people had not started siphoning water from the rivers and streams in the region in 1847, when Mormon settlers led by Brigham Young arrived, the lake would be 3.4 meters higher than it is currently.

Today, three-quarters of the water consumed within the Great Salt Lake watershed is used to irrigate crops, mainly hay that feeds cattle that produce beef or dairy, the state’s chief agricultural commodities.

Mineral extraction from the Great Salt Lake accounts for 9 percent of the water consumed. Companies divert briny water directly from the lake to extract its minerals and produce table salt, fertilizer or magnesium metal. Another 9 percent of the water consumed in the basin gets piped to cities to supply homes and businesses with water for indoor, outdoor and industrial uses. The remaining 8 percent is lost from evaporation from lakes and reservoirs in the basin.

Climate change has also contributed to the Great Salt Lake’s record lows. The worst megadrought in 1,200 years has ensnared the American Southwest for the last 22 years. The lake isn’t refilling fast enough to keep up with the withdrawals upstream, while higher temperatures spur more evaporation from the lake.

How Great Salt Lake water is used

The vast majority of water consumed in the Great Salt Lake watershed goes toward agriculture, with mineral extraction, homes, industries and evaporation accounting for smaller proportions.

Proportion of water consumed for various uses

People use more water to grow crops or keep lawns green when it’s hot and dry, which Patrick Donnelly, a research scientist with the U.S. Fish and Wildlife Service in Missoula, Mont., likens to a climate tax on our waterways. An agricultural producer needs 20 percent more water to grow the same crops in northern Utah than he did 15 years ago, Donnelly says.

Donnelly and colleagues measured changes in 18 saline lakes in the Intermountain West, including the Great Salt Lake, finding an average surface area decrease of 27 percent from 1984 to 2018. The surface area of wetlands in the region diminished by nearly half. The researchers reported in 2020 in Global Change Biology that these losses were driven largely by demands from irrigated agriculture coupled with higher temperatures that increase evaporation.

And the demand for freshwater in is only increasing. Utah is the fastest growing state in the United States, and 80 percent of the people live in the Great Salt Lake watershed.

Donnelly says it’s “unrealistic” to think that the Great Salt Lake region can stretch its water far enough to meet current and increasing demands. He points to a proposal to further dam and divert the Bear River — the tributary that provides over half of the Great Salt Lake’s freshwater and is already riddled with canals, ditches and reservoirs — to supply growing communities in Utah. The project would lower the level of the Great Salt Lake by more than a meter and a half and push salinity to over 22 percent, damaging the lake’s invertebrate community, Wurtsbaugh and Null reported in the book Great Salt Lake Biology, published in 2020.

The consequences of a dry Great Salt Lake

The region is already feeling the consequences of a shrinking lake. Mineral extraction industries are finding it increasingly difficult to get water to their ponds and processing plants. Some ranchers worry they won’t be able to take their full allotment of irrigation water from the lake’s freshwater tributaries. Even Utah’s famous ski resorts feel the impact: An estimated 5 to 10 percent of the fluffy powder that draws millions of tourists to the area’s slopes comes from snowfall triggered by the lake’s relatively warm waters.

If the Great Salt Lake dries completely, the consequences become much more dire. Owens Lake in Central California offers an example of what happens when saline lakes are drained for human water consumption. It went dry in 1926, just 13 years after the Los Angeles Department of Water and Power diverted the Owens River into a 375-kilometer-long aqueduct to supply water to Southern California. This engineering feat turned fertile farmlands into an eerie dust bowl and caused health problems for residents in nearby communities.

Once a lake bed is exposed, winds kick up ferocious dust storms. Those windblown sediments contribute to air pollution and can contribute to asthma, lung cancer and cardiopulmonary disease, among other health issues. Owens Lake has been among the largest sources of dust pollution in the nation. The City of Los Angeles has spent more than $2.5 billion mitigating the dust through projects at the lake bed such as shallow flooding, seeding and planting vegetation, spreading gravel or tilling the ground.

This is sobering news for Salt Lake City, which had at least seven times as much lake bed exposed last fall as Owens Lake. “In the summers when we get a strong south wind, you can literally watch a cloud of dust coming off the lake where the lake bed has been exposed because of the low receding water,” Yamane says.

Sediments at the bottom of saline lakes also collect a slew of pollutants from human activities, such as chemicals from urban and agricultural runoff and toxic mine waste. Historical mining and smelting around the Great Salt Lake, for example, deposited heavy metals like mercury and lead that have accumulated in the lake bed’s sediments. Once exposed, these metals can be transported by dust particles and may increase rates of disease associated with air pollution, according to the January report from the BYU team.

The drying is also bad for birds. The Great Salt Lake supports nearly 350 different species, many of them migratory birds that use the lake to rest and refuel as they cruise north or south along the Central and Pacific flyways. While many migrating birds can make do with any lake, fresh or salty, some species are saline specialists, like eared grebes and phalaropes.

According to a 2017 report by the National Audubon Society, more than half of the saline lakes in the West that are most important for birds have shrunk by 50 to 95 percent in their surface area over the last 150 years. “The birds are competing now for a much more limited resource than they did before,” says John Luft, manager of the Great Salt Lake Ecosystem Program at the Utah Division of Wildlife Resources.

During some fall migrations, 5 million eared grebes — perhaps 95 percent of the species’ entire population — have stopped over in Utah, Luft says. Each grebe needs to eat up to 30,000 brine shrimp per day from the Great Salt Lake before continuing on its long migratory journey. Brine shrimp flourish between 12 and 17 percent salinity, but they start to decline drastically once salinity passes that threshold — and the south arm of the Great Salt Lake reached 18 percent last September. (For comparison, the ocean averages about 3.5 percent salinity.)

What to do about Great Salt Lake’s drying?

While Wurtsbaugh says that “drying is definitely the projection and the worry,” he believes residents can help halt the decline by using less water.

Null agrees. “We can save Great Salt Lake, but it’s a long solution, not something that we’re going to fix overnight.” The most important way to make progress, she says, is for people to care enough to change their behavior.

Utahans have the second highest per capita use of water in the country. Encouraging people to use less water is the simplest — and cheapest — way to begin refilling the Great Salt Lake. This includes using rain barrels, permeable pavement or xeriscaping in urban or suburban areas, as well as improving the efficiency of irrigation structures on farms and ranches.

Permanently mandating water cutbacks would cost between $5 to $32 per person, according to Wurtsbaugh and Null’s research. The BYU team’s report, coauthored by 32 Great Salt Lake experts, emphasizes that conservation is not only the most cost-effective response, but also “the only way to provide adequate water in time to save Great Salt Lake.” Specifically, the report stresses the importance of reducing outdoor water use, especially on agricultural fields.

Last year, faced with withering streams and ominous lake levels, the Utah State Legislature passed a slew of bills focused on voluntary water conservation. “Honestly, I look at this as our time to shine,” says Joel Ferry, executive director of the Utah Department of Natural Resources. “The people of Utah … want to solve the problem.”

Some of the recent legislation calls for obvious fixes — like banning any requirements that residents water their lawns. Before this legislation, fines for not keeping your grass green were common around Salt Lake City. Other laws focus on making sure basic systems are in place to track water use, from the individual to the watershed level.

For instance, the legislature allocated $250 million to install tens of thousands of meters to track people’s outdoor water use. According to Wurtsbaugh, a previous effort to install meters in the Weber River, which flows into the Great Salt Lake, showed that “just knowing how much water you’re using compared to what your neighbor’s using” meant people used about 25 percent less water. Metering also opens the door to charging people accurately — or more — for the water they use. Currently, most residents pay a flat fee for all outdoor water, regardless of whether they are watering a few flowers or filling an entire swimming pool.

Since agriculture accounts for three-quarters of the water use in the Great Salt Lake region, Utah is also encouraging farmers and ranchers to conserve water. This shift has the advantage of making operations more drought resilient, says Ferry, who is also a fifth-generation rancher who irrigates with water from the Bear River. “Producers want to be part of the solution. They have to be.”

In 2022, the Utah Department of Agriculture and Food’s Water Optimization Program granted $70 million to help farmers and ranchers install drip or sprinkler systems that use less water. The Utah State Legislature also appropriated $40 million last year to preserve flows to the lake. Much of that is dedicated to setting up a water trust to lease water rights from farmers or ranchers willing to sell. Essentially, a water trust would pay irrigators to leave some or all of the water they are permitted to use in the stream or river instead of diverting it to water crops. These water leases could last a single summer or several years.

Water leasing is helping restore other ailing saline lakes. In Nevada, purchasing water rights from willing sellers has boosted the level of Walker Lake, about 150 kilometers southeast of Reno. But because water is so valuable in the arid West, it’s an expensive way to refill a lake. The Walker Basin Restoration Program has spent at least $92 million but has only acquired 53 percent of the water needed to support native fish and wildlife in the lake.

Big infrastructure and Mother Nature

If water conservation programs don’t work, Utahans might be faced with a more extreme solution to save the Great Salt Lake.

The alternative to conservation measures, says Null, is “hanging our hopes on big new infrastructure projects” that attempt to refill the lake by bringing water from other basins. One example is the Central Utah Project, which has been in the works for more than 80 years and is still incomplete. This project pipes up to 310 billion liters of water from the Colorado River Basin in eastern Utah, infamously over-tapped already, into a series of reservoirs and tunnels to supply water for irrigation, municipal and industrial uses in the Great Salt Lake’s watershed. According to the U.S. Bureau of Reclamation, the initial plans for the Central Utah Project put it “among the most complex” water resources development projects ever undertaken by the bureau, estimated to cost $3 billion once all phases are complete.

In the Aral Sea, dikes were built to preserve the remaining sliver of wet habitat, which meant permanently sacrificing the rest of the historical lake area. In California, Mono Lake was saved by a lawsuit based on the public trust doctrine, which says that the government has a responsibility to protect the resources that belong to everyone. The court’s decision cut off water to cities in Southern California that held water rights to the river that fed Mono Lake. Los Angeles made up the difference in part through water conservation measures.

Ferry says Utah isn’t closing the door on any options. Some of the more unorthodox ideas floating around Salt Lake City include cloud seeding to boost precipitation and piping water from the Pacific Ocean to refill the lake. But the most common strategy echoed by the Utahans interviewed for this story: Pray for snow.

“I’m very optimistic that we are at the lowest of lows,” Ferry said last fall, “as long as we get some snow.”

Utah received a record-setting amount of snow this winter, so the situation is looking up … at least for now. But it will take several years of above-average precipitation to reverse Utah’s lingering drought.

“Mother Nature has a huge role to play in this,” Yamane says. “And it’s going to take a mind-set change, a cultural change and policy changes. But along with Mother Nature, we should be able to save it.”


"""

doc4 = Document(page_content=source4, metadata={"topic": "environment"})

source5 = """
The European Union (EU) put forward an ambitious climate mitigation target of reducing greenhouse gas (GHG) emissions by at least 55% below 1990 levels by 2030. How will the EU deliver? Carbon pricing – through the European Union Emissions Trading System (EU ETS) – is expected to deliver a large part of the emissions reductions. Under an ETS, installation operators can trade GHG emission permits with each other, ensuring that emissions are reduced cost-effectively. Launched in 2005, the EU ETS is the world’s first international ETS, covering over 14,000 energy-intensive plants across 30 European countries, accounting for around 40% of the EU’s total GHG emissions. From the outset the EU ETS raised concerns about its environmental effectiveness and potential negative economic effects for the European industry by putting regulated firms at a disadvantage vis-a-vis their foreign competitors.

A recent paper ‘The joint impact of the European Union emissions trading system on carbon emissions and economic performance’ published by OECD authors in the leading Journal of Environmental Economics and Management sheds light on this concern. Based on an earlier OECD working paper, the study is the first comprehensive, European-wide analysis of the impact of the EU ETS on both carbon emissions and economic performance of regulated companies during the first two phases of the system’s existence, from 2005 to 2012.

The study uses data for carbon emissions of installations from the national Pollutant Release and Transfer Registers (PRTR) of France, Netherlands, Norway and the United Kingdom, complemented with data from the European PRTR. It also includes economic data of firms for all European countries to investigate the impact of EU ETS on various economic dimensions, including employment, fixed assets, profits, and revenues. The study makes use of the EU ETS inclusion criteria, according to which installations below a certain capacity threshold do not need to participate in the carbon market. It compares installations or firms operating in the same country and the same sector and of similar characteristics, but which fall under different regulatory regimes since the launch of the EU ETS.

So what does the study tell us?

The EU ETS reduced emissions while not negatively affecting economic outcomes

The EU ETS led to a reduction of carbon emissions of around 10% between 2005 and 2012 but has not had any adverse impact on employment (see Figure 1). Most of the emissions reductions were observed in the second trading phase of the EU ETS and were primarily driven by larger installations. This is in line with the observation that pollution control technologies are capital-intensive and involve relatively high fixed costs. There is also evidence that a more generous allocation of free allowances results in a weaker reduction of emissions.

Figure 1. The impact of the EU ETS on jobs and CO2 emissions

Note: The graph shows the percentage change on CO2 emissions and number of employees by year of firms participating in the EU ETS versus those not participating.

Source: Based on Dechezleprêtre et al. (2018) and Dechezleprêtre et al. (2023)

The study also finds that the EU ETS has not had a negative effect on regulated firms’ revenue, profits, fixed assets and jobs. In fact, the EU ETS seemed to have led to an increase of revenues and fixed assets of regulated firms – contrary to what could have been expected. One explanation could be that the EU ETS induced regulated firms to increase investment – likely in carbon-saving technologies – which, in turn, may have increased productivity.

More research is needed to reflect more recent developments in carbon pricing

In its first eight years of existence, the EU ETS effectively reduced carbon emissions without negatively affecting the economic performance and competitiveness of European regulated firms. This is in line with recent literature reviews on the effects of carbon pricing on environmental and economic outcomes. While these results demonstrate that concerns about negative effects of the EU ETS on the competitiveness of the European industry have been vastly overplayed, more research is needed to assess these findings against new realities. In fact, the period between 2005-2012 was characterised by relatively low permit prices of EUR 20/tCO2 on average and a generous allocation of free allowances. From mid-2020, permit prices were fluctuating around EUR 80t/CO2, so it remains to be seen whether these findings hold true in a high price environment.
"""

doc5 = Document(page_content=source5, metadata={"topic": "emissions"})


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