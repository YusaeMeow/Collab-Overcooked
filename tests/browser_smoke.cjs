// Optional real-browser integration check. Start the default human_play server.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.BROWSER_CHANNEL ? { channel: process.env.BROWSER_CHANNEL } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1060 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.argv[2] || 'http://127.0.0.1:8080');
    await page.waitForFunction(() => document.getElementById('connection').textContent.includes('Player'));
    const oldEpisode = await page.evaluate(() => state.episode_id);
    await page.locator('#reset').click();
    await page.waitForFunction(old => state.episode_id !== old && state.paused &&
      state.state.timestep === 0, oldEpisode);
    await page.locator('#start').click();
    await page.waitForFunction(() => !state.paused);
    await page.locator('#game').focus();
    const move = async (key, x, y) => {
      await page.keyboard.down(key);
      await page.waitForFunction(([x, y]) => state.state.players[1].position[0] === x &&
        state.state.players[1].position[1] === y, [x, y]);
      await page.keyboard.up(key);
    };
    const face = async (key, dx, dy) => {
      await page.keyboard.down(key);
      await page.waitForFunction(([dx, dy]) => state.state.players[1].orientation[0] === dx &&
        state.state.players[1].orientation[1] === dy, [dx, dy]);
      await page.keyboard.up(key);
    };
    await move('ArrowUp', 1, 1);
    await face('ArrowLeft', -1, 0);
    await page.keyboard.press('Space');
    await page.waitForFunction(() => state.state.players[1].held_object?.name === 'egg');
    await move('ArrowDown', 1, 2);
    await face('ArrowRight', 1, 0);
    await page.keyboard.press('Space');
    await page.waitForFunction(() => state.done, null, { timeout: 20000 });
    assert.equal(await page.locator('#score').innerText(), '20');
    assert.equal(await page.locator('#mode').innerText(), 'Finished');
    assert.deepEqual(errors, []);
    if (process.env.SCREENSHOT_PATH) await page.screenshot({ path: process.env.SCREENSHOT_PATH, fullPage: true });
    console.log('PASS: browser keyboard handoff, agent cooking, serving, and no JavaScript errors');
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    console.log('PASS: mobile viewport has no horizontal overflow');
    const finishedEpisode = await page.evaluate(() => state.episode_id);
    await page.locator('#reset').click();
    await page.waitForFunction(old => state.episode_id !== old && state.paused, finishedEpisode);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
