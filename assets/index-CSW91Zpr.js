(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))r(t);new MutationObserver(t=>{for(const i of t)if(i.type==="childList")for(const a of i.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&r(a)}).observe(document,{childList:!0,subtree:!0});function s(t){const i={};return t.integrity&&(i.integrity=t.integrity),t.referrerPolicy&&(i.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?i.credentials="include":t.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function r(t){if(t.ep)return;t.ep=!0;const i=s(t);fetch(t.href,i)}})();const o={assets:[{tag:"NVDA",name:"NVIDIA Corporation",color:"#76b900",currentPrice:208.27,expectedReturn:1.03,volatility:"Moderate",sentiment:.85,confidenceInterval:[201.1,220.19]},{tag:"USO",name:"United States Oil Fund",color:"#eb6e2b",currentPrice:99.13,expectedReturn:-.18,volatility:"High Swing/Risk",sentiment:-.42,confidenceInterval:[90,108.8]},{tag:"EGX",name:"EGX70 EWI Index",color:"#FFCE00",currentPrice:13819.4,expectedReturn:1.18,volatility:"Low Risk/Stable",sentiment:.62,confidenceInterval:[13770.07,14200.36]}],impactNews:[{asset:"NVDA",date:"2025-07-09",headline:"Nvidia Becomes First Company To Reach $4T Market Cap",sentiment:1,reasoning:"Nvidia reaching a $4T market cap signifies extremely positive investor sentiment and strong market performance. This indicates increased confidence in the company's future growth prospects, driven by its leading position in AI and data center technologies.",priceImpact:"+4.2% in 24h"},{asset:"USO",date:"2020-04-20",headline:"Crude Oil Futures Settle At -$37.63/Bbl, Down 305.97%",sentiment:-1,reasoning:"The reported settlement price of -$37.63/bbl indicates an extreme oversupply situation and a collapse in demand. This is a historically unprecedented event, suggesting significant storage constraints and a lack of buyers.",priceImpact:"-305% (Liquidity Crisis)"},{asset:"NVDA",date:"2026-04-27",headline:"NVIDIA Becomes The First Company To Close Above A $5T Market Cap",sentiment:1,reasoning:"NVIDIA surpassing a $5T market cap signifies extremely positive investor sentiment and strong market confidence in the company's future growth prospects, driven by its dominance in the AI hardware market.",priceImpact:"+5.8% in 24h"}]};document.addEventListener("DOMContentLoaded",()=>{c(),d(),l(),p()});function c(){const n=document.querySelectorAll("nav li"),e=document.querySelectorAll(".section");n.forEach(s=>{s.addEventListener("click",()=>{const r=s.getAttribute("data-section");n.forEach(t=>t.classList.remove("active")),s.classList.add("active"),e.forEach(t=>{t.classList.remove("active"),t.id===r&&t.classList.add("active")})})})}function d(){const n=document.getElementById("asset-grid");n.innerHTML=o.assets.map(e=>`
        <div class="asset-card glass">
            <div class="asset-header">
                <div>
                    <span class="asset-tag" style="color: ${e.color}">${e.tag}</span>
                    <p class="asset-name">${e.name}</p>
                </div>
                <div class="asset-status">AI ACTIVE</div>
            </div>
            <div class="price-main">$${e.currentPrice.toLocaleString()}</div>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span>Expected Return (5d)</span>
                    <strong class="${e.expectedReturn>=0?"pos":"neg"}">
                        ${e.expectedReturn>=0?"+":""}${e.expectedReturn}%
                    </strong>
                </div>
                <div class="metric-box">
                    <span>Volatility Risk</span>
                    <strong>${e.volatility}</strong>
                </div>
            </div>
            <div class="sentiment-meter">
                <div class="sentiment-fill" style="width: ${(e.sentiment+1)*50}%; background-color: ${e.color}; color: ${e.color}"></div>
            </div>
            <p style="font-size: 0.7rem; color: #94a3b8; margin-top: 10px; text-align: center;">
                Sentiment Analysis: ${Math.round(e.sentiment*100)}% Confidence
            </p>
        </div>
    `).join("")}function l(){const n=document.getElementById("pulse-list");n.innerHTML=o.impactNews.map(e=>{const s=o.assets.find(r=>r.tag===e.asset);return`
            <div class="pulse-item">
                <div class="pulse-date">${e.date}</div>
                <div class="pulse-body">
                    <h4>${e.headline}</h4>
                    <p class="pulse-reasoning">${e.reasoning}</p>
                    <div class="pulse-meta">
                        <span class="tag" style="border: 1px solid ${s.color}; color: ${s.color}">${e.asset}</span>
                        <span class="tag" style="background: ${e.sentiment>0?"#00ff8822":"#ff4a4a22"}; color: ${e.sentiment>0?"#00ff88":"#ff4a4a"}">
                            AI SCORE: ${e.sentiment>0?"+":""}${e.sentiment}
                        </span>
                        <span class="tag" style="color: white; background: rgba(255,255,255,0.1)">IMPACT: ${e.priceImpact}</span>
                    </div>
                </div>
            </div>
        `}).join("")}function p(){const n=document.getElementById("equity-chart");n.innerHTML=`
        <svg viewBox="0 0 800 300" style="width: 100%; height: 100%;">
            <!-- Benchmark Curve (Gray) -->
            <path d="M0,250 L100,240 L200,260 L300,230 L400,245 L500,210 L600,225 L700,200 L800,210" 
                  fill="none" stroke="#444" stroke-width="2" stroke-dasharray="5,5" />
            <!-- Our Hybrid Curve (Green) -->
            <path d="M0,250 L100,230 L200,220 L300,180 L400,160 L500,120 L600,110 L700,80 L800,60" 
                  fill="none" stroke="#76b900" stroke-width="4" />
            <text x="10" y="20" fill="#76b900" font-size="12" font-weight="bold">Hybrid Sentiment Strategy</text>
            <text x="10" y="40" fill="#666" font-size="12">Buy & Hold Benchmark</text>
        </svg>
    `}
