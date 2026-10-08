export interface NewsItem {
  id: number;
  title: string;
  text: string;
  label: 'REAL' | 'FAKE';
  source?: string;
}

export const DATASET: NewsItem[] = [
  {
    id: 1,
    title: "NASA James Webb Space Telescope discovers oldest known galaxy in universe",
    text: "Astronomers using the James Webb Space Telescope have identified the most distant and oldest galaxy ever observed, designated JADES-GS-z14-0. The discovery was confirmed through spectroscopic analysis and published in peer-reviewed astronomical journals by international research teams.",
    label: "REAL",
    source: "NASA / Space Science"
  },
  {
    id: 2,
    title: "SHOCKING: Secret government cure for all cancers discovered in lemon peels!",
    text: "Doctors and pharmaceutical companies are terrified! An anonymous whistleblower has revealed that boiling lemon peels with baking soda cures all stage 4 cancers in 48 hours. Big Pharma is desperately trying to delete this post from the internet! Share before it gets taken down!",
    label: "FAKE",
    source: "Viral Social Media Hoax"
  },
  {
    id: 3,
    title: "Federal Reserve holds benchmark interest rate steady amid cooling inflation",
    text: "The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate between 5.25% and 5.50%. Federal Reserve Chairman Jerome Powell stated that committee members are monitoring ongoing employment figures and consumer price index trends before considering future policy rate reductions.",
    label: "REAL",
    source: "Financial News / Reuters"
  },
  {
    id: 4,
    title: "BREAKING: Pope Francis endorses Donald Trump for US President in unprecedented move",
    text: "In a stunning development that has shocked the Vatican, Pope Francis has officially endorsed Donald Trump for President of the United States. In a secret statement released to independent patriotic blogs, the Pontiff claimed divine intervention guided his decision.",
    label: "FAKE",
    source: "Political Disinformation Blog"
  },
  {
    id: 5,
    title: "World Health Organization releases updated guidelines on cardiovascular health",
    text: "The World Health Organization (WHO) has issued new comprehensive recommendations aimed at reducing global mortality from cardiovascular diseases. The guidance emphasizes balanced dietary patterns, reduced sodium intake, regular moderate physical activity, and early hypertension screening in primary healthcare settings.",
    label: "REAL",
    source: "WHO Official Bulletin"
  },
  {
    id: 6,
    title: "ALERT: Drinking boiled garlic water cures 100% of viral infections overnight",
    text: "A top military doctor has leaked the ancient natural formula that completely destroys all coronavirus and influenza viruses within six hours. Boil eight cloves of raw garlic with honey and drink immediately. Hospitals are covering this up to keep ICU beds occupied!",
    label: "FAKE",
    source: "WhatsApp Medical Forward"
  },
  {
    id: 7,
    title: "European Space Agency successfully launches Euclid satellite to map dark universe",
    text: "The European Space Agency (ESA) Euclid space mission successfully launched aboard a SpaceX Falcon 9 rocket from Cape Canaveral. Euclid aims to construct the largest and most accurate 3D map of the universe, observing billions of galaxies across 10 billion light-years to investigate dark matter and dark energy.",
    label: "REAL",
    source: "ESA Press Release"
  },
  {
    id: 8,
    title: "5G cellular towers confirmed to be emitting radiation that controls human thoughts",
    text: "Declassified top secret documents reveal that 5G cellular communication antennas are equipped with micro-frequency transmitters designed to manipulate brainwaves of citizens in major metropolitan areas. Protect your family by wrapping wireless routers in heavy aluminum foil!",
    label: "FAKE",
    source: "Conspiracy Forum"
  },
  {
    id: 9,
    title: "Global renewable energy capacity expanded by 50% in 2023, reports International Energy Agency",
    text: "According to the latest annual market report from the International Energy Agency (IEA), global renewable capacity additions reached 510 gigawatts in 2023, representing a 50% increase compared to the previous year. Solar photovoltaic systems accounted for three-quarters of the worldwide expansion.",
    label: "REAL",
    source: "IEA Market Report"
  },
  {
    id: 10,
    title: "URGENT: Government installing microchips in standard drinking tap water supplies",
    text: "A heroic chemical lab technician has posted video evidence showing self-assembling nano-bots flowing through municipal tap water networks. The chips sync with satellite networks to monitor private citizens in their living rooms. Share this immediately before social media bans our account!",
    label: "FAKE",
    source: "Viral Telegram Rumor"
  },
  {
    id: 11,
    title: "Bank of England cuts key interest rate following steady drop in inflation",
    text: "The Bank of England reduced borrowing costs by 25 basis points to 5.0% following a vote by the Monetary Policy Committee. Governor Andrew Bailey noted that while inflation has returned close to the 2% target, policymakers remain vigilant against persistent domestic price pressures.",
    label: "REAL",
    source: "Bank of England"
  },
  {
    id: 12,
    title: "Celebrity billionaire leaves entire fortune to random followers who retweet this post",
    text: "Tech entrepreneur Elon Musk announced today that he will give ten million dollars each to the first 50,000 users who repost this link, follow this Telegram channel, and verify their credit card details. This giveaway has been verified by independent blockchain auditors!",
    label: "FAKE",
    source: "Social Media Phishing Post"
  },
  {
    id: 13,
    title: "Researchers develop ultra-efficient perovskite solar cell achieving record efficiency",
    text: "Materials scientists at Oxford University and partnering laboratories have synthesized a tandem perovskite-silicon solar cell demonstrating over 33% power conversion efficiency in verified test conditions. The findings were documented in the journal Science.",
    label: "REAL",
    source: "Science Journal"
  },
  {
    id: 14,
    title: "Ancient Himalayan monks reveal breathing technique to live without food for 12 years",
    text: "Himalayan holy men have broken centuries of silence to share the miracle breathwork that eliminates the physical biological need for eating food. Practicing this 10-minute daily exercise draws pure cosmic solar energy directly into the bloodstream, curing obesity and aging forever!",
    label: "FAKE",
    source: "Holistic Health Blog"
  },
  {
    id: 15,
    title: "India successfully lands Chandrayaan spacecraft on lunar south polar region",
    text: "The Indian Space Research Organisation (ISRO) achieved an unprecedented soft touchdown of the Chandrayaan-3 lander module near the south pole of the Moon, making India the fourth nation to accomplish a soft lunar landing and the first near the southern polar latitude.",
    label: "REAL",
    source: "ISRO / Science News"
  },
  {
    id: 16,
    title: "LEAKED: The Earth is actually flat and NASA guards giant Antarctic ice wall",
    text: "High-ranking naval navigators have leaked photographic evidence proving that Antarctica is not a continent, but a 300-foot ice barrier retaining the oceans on an endless flat plane. International treaties forbid civilian flights across the southern rim to hide the real perimeter!",
    label: "FAKE",
    source: "Flat Earth Conspiracy"
  },
  {
    id: 17,
    title: "Supreme Court delivers ruling on regulatory authority of administrative federal agencies",
    text: "The Supreme Court of the United States delivered an opinion in a major administrative law dispute, overturning historical judicial deference doctrines and clarifying the statutory standards under which federal courts must interpret legislative enactments.",
    label: "REAL",
    source: "Legal Times / AP"
  },
  {
    id: 18,
    title: "Shocking discovery: Eating bananas past 6 PM turns stomach acids into toxic poison",
    text: "Nutritional experts warn that the natural potassium in ripe bananas undergoes dangerous chemical mutations after sundown, causing severe liver inflammation and toxic bacterial growth. Never consume yellow fruit after dusk!",
    label: "FAKE",
    source: "Clickbait Diet Site"
  },
  {
    id: 19,
    title: "Nobel Prize in Chemistry awarded for computational protein design and structure prediction",
    text: "The Royal Swedish Academy of Sciences awarded the Nobel Prize in Chemistry to scientists recognized for their pioneering contributions to computational protein design and machine learning-driven protein structure determination, revolutionizing biochemical research and pharmaceutical drug discovery.",
    label: "REAL",
    source: "Nobel Prize Committee"
  },
  {
    id: 20,
    title: "Miracle fruit found in Amazon rainforest dissolves 45 pounds of belly fat in 7 days",
    text: "Harvard researchers are stunned after clinical trials showed that a rare violet Amazonian berry burns body fat 800% faster than any gym workout. No diet changes required! Supplies are strictly limited due to intense pressure from greedy pharmaceutical executives!",
    label: "FAKE",
    source: "Affiliate Scam Webpage"
  }
];
