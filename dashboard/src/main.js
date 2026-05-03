import { marketData } from './data.js';

document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for navigation links
    initSmoothScrolling();
    
    // Render dynamic content
    renderAssetCards();
    renderPulseNews();
    renderBacktestChart();
});

function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                // Account for the fixed top nav (80px + 20px padding)
                const offsetTop = targetSection.getBoundingClientRect().top + window.pageYOffset - 100;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

function renderAssetCards() {
    const grid = document.getElementById('asset-grid');
    grid.innerHTML = marketData.assets.map(asset => `
        <div class="asset-card glass">
            <div class="asset-header">
                <div>
                    <span class="asset-tag" style="color: ${asset.color}">${asset.tag}</span>
                    <p class="asset-name">${asset.name}</p>
                </div>
                <div class="asset-status">AI ACTIVE</div>
            </div>
            <div class="price-main">$${asset.currentPrice.toLocaleString()}</div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span>Expected Return (5d)</span>
                    <strong class="${asset.expectedReturn >= 0 ? 'pos' : 'neg'}">
                        ${asset.expectedReturn >= 0 ? '+' : ''}${asset.expectedReturn}%
                    </strong>
                </div>
                <div class="metric-box">
                    <span>Volatility Risk</span>
                    <strong>${asset.volatility}</strong>
                </div>
            </div>
            <div class="sentiment-meter">
                <div class="sentiment-fill" style="width: ${(asset.sentiment + 1) * 50}%; background-color: ${asset.color}; color: ${asset.color}"></div>
            </div>
            <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 12px; text-align: center;">
                Sentiment Analysis: ${Math.round(asset.sentiment * 100)}% Confidence
            </p>
        </div>
    `).join('');
}

function renderPulseNews() {
    const list = document.getElementById('pulse-list');
    list.innerHTML = marketData.impactNews.map(news => {
        const asset = marketData.assets.find(a => a.tag === news.asset);
        return `
            <div class="pulse-item">
                <div class="pulse-date">${news.date}</div>
                <div class="pulse-body">
                    <h4>${news.headline}</h4>
                    <p class="pulse-reasoning">${news.reasoning}</p>
                    <div class="pulse-meta">
                        <span class="tag" style="border: 1px solid ${asset.color}; color: ${asset.color}">${news.asset}</span>
                        <span class="tag" style="background: ${news.sentiment > 0 ? '#00ff8822' : '#ff4a4a22'}; color: ${news.sentiment > 0 ? '#00ff88' : '#ff4a4a'}">
                            AI SCORE: ${news.sentiment > 0 ? '+' : ''}${news.sentiment}
                        </span>
                        <span class="tag" style="color: white; background: rgba(255,255,255,0.1)">IMPACT: ${news.priceImpact}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function renderBacktestChart() {
    const container = document.getElementById('equity-chart');
    // Simple SVG mock of a winning strategy curve
    container.innerHTML = `
        <svg viewBox="0 0 800 300" style="width: 100%; height: 100%; display: block;">
            <!-- Benchmark Curve (Gray) -->
            <path d="M0,250 L100,240 L200,260 L300,230 L400,245 L500,210 L600,225 L700,200 L800,210" 
                  fill="none" stroke="#444" stroke-width="2" stroke-dasharray="5,5" />
            <!-- Our Hybrid Curve (Green) -->
            <path d="M0,250 L100,230 L200,220 L300,180 L400,160 L500,120 L600,110 L700,80 L800,60" 
                  fill="none" stroke="#76b900" stroke-width="4" />
            <text x="10" y="20" fill="#76b900" font-size="14" font-weight="bold">Hybrid Sentiment Strategy</text>
            <text x="10" y="40" fill="#94a3b8" font-size="14">Buy & Hold Benchmark</text>
        </svg>
    `;
}
