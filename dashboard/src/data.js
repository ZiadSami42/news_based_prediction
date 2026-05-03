export const marketData = {
    assets: [
        {
            tag: 'NVDA',
            name: 'NVIDIA Corporation',
            color: '#76b900',
            currentPrice: 208.27,
            expectedReturn: 1.03,
            volatility: 'Moderate',
            sentiment: 0.85,
            confidenceInterval: [201.10, 220.19]
        },
        {
            tag: 'USO',
            name: 'United States Oil Fund',
            color: '#eb6e2b',
            currentPrice: 99.13,
            expectedReturn: -0.18,
            volatility: 'High Swing/Risk',
            sentiment: -0.42,
            confidenceInterval: [90.00, 108.80]
        },
        {
            tag: 'EGX',
            name: 'EGX70 EWI Index',
            color: '#FFCE00',
            currentPrice: 13819.40,
            expectedReturn: 1.18,
            volatility: 'Low Risk/Stable',
            sentiment: 0.62,
            confidenceInterval: [13770.07, 14200.36]
        }
    ],
    impactNews: [
        {
            asset: 'NVDA',
            date: '2025-07-09',
            headline: 'Nvidia Becomes First Company To Reach $4T Market Cap',
            sentiment: 1.0,
            reasoning: "Nvidia reaching a $4T market cap signifies extremely positive investor sentiment and strong market performance. This indicates increased confidence in the company's future growth prospects, driven by its leading position in AI and data center technologies.",
            priceImpact: '+4.2% in 24h'
        },
        {
            asset: 'USO',
            date: '2020-04-20',
            headline: 'Crude Oil Futures Settle At -$37.63/Bbl, Down 305.97%',
            sentiment: -1.0,
            reasoning: "The reported settlement price of -$37.63/bbl indicates an extreme oversupply situation and a collapse in demand. This is a historically unprecedented event, suggesting significant storage constraints and a lack of buyers.",
            priceImpact: '-305% (Liquidity Crisis)'
        },
        {
            asset: 'NVDA',
            date: '2026-04-27',
            headline: 'NVIDIA Becomes The First Company To Close Above A $5T Market Cap',
            sentiment: 1.0,
            reasoning: "NVIDIA surpassing a $5T market cap signifies extremely positive investor sentiment and strong market confidence in the company's future growth prospects, driven by its dominance in the AI hardware market.",
            priceImpact: '+5.8% in 24h'
        }
    ]
};
