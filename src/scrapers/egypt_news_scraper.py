import asyncio
import datetime
import json
import os
from playwright.async_api import async_playwright

# --- Configuration ---
SAVE_DIR = "egypt_news_scraped_json_files"
os.makedirs(SAVE_DIR, exist_ok=True)
START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2026, 4, 26)

async def safe_goto(page, url, retries=3, timeout=60000):
    """Handles unstable connections with retries and exponential backoff."""
    for i in range(retries):
        try:
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            return True
        except Exception as e:
            wait_time = 5 * (i + 1)
            print(f"  [Attempt {i+1}] Connection error for {url}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    return False

async def get_day_links(page, date_obj):
    """Discovers article URLs for a specific day, filtering out noise."""
    end_date = date_obj + datetime.timedelta(days=1)
    url = f"https://www.zawya.com/en/search?q=Egypt&articleType=NEWS&from={date_obj.isoformat()}Z&to={end_date.isoformat()}Z"

    if not await safe_goto(page, url):
        return []

    try:
        # Wait for the results to actually appear
        await page.wait_for_selector(".teaser", timeout=15000)
    except:
        return []

    valid_links = await page.evaluate('''() => {
        const links = [];
        document.querySelectorAll('.teaser').forEach(t => {
            const dateText = t.querySelector('.teaser-published-date')?.innerText;
            const linkTag = t.querySelector('a');
            // Ensure we only get items that actually have a date (ignoring sidebar noise)
            if (dateText && linkTag && linkTag.href) {
                links.push(linkTag.href);
            }
        });
        return links;
    }''')
    return list(set(valid_links))

async def extract_article_data(page, url):
    """Navigates to the article and performs deep, clean data extraction."""
    if not await safe_goto(page, url):
        return None

    try:
        await page.wait_for_selector(".article-body", timeout=15000)
        
        article_data = await page.evaluate('''() => {
            // 1. Clean the DOM of disclaimers/copyrights
            const noise = document.querySelectorAll('.syndigate_disclaimer, .footnote-disclaimer');
            noise.forEach(el => el.remove());

            // 2. Extract structured fields
            const title = document.querySelector('.article-title')?.innerText?.trim();
            const lead = document.querySelector('.article-lead')?.innerText?.trim();
            const date = document.querySelector('.article-date span')?.innerText?.trim();
            
            const author = document.querySelector('.author-name-text')?.innerText?.trim();
            const providerRaw = document.querySelector('.provider')?.innerText?.trim() || '';
            const provider = providerRaw.replace(/^,\\s*/, ''); 

            const category = document.querySelector('.article-keyword .keyword-text')?.innerText?.trim();
            const tags = Array.from(document.querySelectorAll('.related-topics a')).map(a => a.innerText.trim());

            // 3. Join clean body paragraphs
            const paragraphs = Array.from(document.querySelectorAll('.article-body p'))
                .map(p => p.innerText.trim())
                .filter(text => text.length > 0);
            
            return {
                title, lead, category, date, author, provider, tags,
                body: paragraphs.join('\\n\\n'),
                url: document.location.href
            };
        }''')
        return article_data
    except Exception as e:
        print(f"  Error parsing content for {url}: {e}")
        return None

async def main():
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = await context.new_page()

        current_date = START_DATE

        while current_date < END_DATE:
            date_str = current_date.strftime('%Y-%m-%d')
            file_path = os.path.join(SAVE_DIR, f"{date_str}.json")

            # Checkpoint: Skip if today's work is already done
            if os.path.exists(file_path):
                print(f"Skipping {date_str} (File exists)")
                current_date += datetime.timedelta(days=1)
                continue

            print(f"--- Processing {date_str} ---")
            urls = await get_day_links(page, current_date)
            print(f"Found {len(urls)} valid URLs.")

            day_results = []
            for url in urls:
                print(f"  Scraping: {url}")
                content = await extract_article_data(page, url)
                if content:
                    day_results.append(content)
                
                # Polite delay between articles
                await asyncio.sleep(1.2)

            # Save incrementally for this day
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(day_results, f, indent=4, ensure_ascii=False)

            # Daily batch delay and increment
            current_date += datetime.timedelta(days=1)
            await asyncio.sleep(2)

        await browser.close()
        print("Scraping Job Complete.")

if __name__ == "__main__":
    asyncio.run(main())