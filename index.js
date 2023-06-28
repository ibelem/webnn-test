import { chromium } from 'playwright';
import { readFile } from 'fs/promises';
const json = JSON.parse(
  await readFile(
    new URL('./config.json', import.meta.url)
  )
);

const cpu = json.enviroment.cpu, ram = json.enviroment.ram, gpu = json.enviroment.gpu, os = json.enviroment.os;
const browsername = json.browser.name, browserbuild = json.browser.build;

let browser, context, page, backend;

const setup = async () => {

  console.log(`Starting testing...`);
  console.log(`----------------------------------------`);
  console.log(`CPU: ${cpu}`);
  console.log(`RAM: ${ram}`);
  console.log(`GPU: ${gpu}`);
  console.log(`OS: ${os}`);
  console.log(`Browser: ${browsername}, Build: ${browserbuild}`);
  console.log(`----------------------------------------`);

  browser = await chromium.launch({
    headless: false,
    executablePath: json.browser.path,
    args: [json.browser.flags]
  });

  context = await browser.newContext();
  page = await context.newPage();
  await page.goto(json.url);

}

const teardown = async () => {
  await context.close();
  await browser.close();

  console.log(`----------------------------------------`);
  console.log(`Testing completed`);
}

const main = async () => {
  await setup();
  for (let i of json.models) {
    let category = i.category, model = i.model, url = i.url;
    console.log(`Category: ${category}, Model: ${model}, URL: ${url}`);

    for (let backend of json.backends) {
      await run(url, backend);
    }
  }
  await teardown(context, browser);
}

main();

const run = async (url, backend) => {
  page.reload();
  const models = page.locator('#lil-gui-name-1');
  await models.waitFor({ state: 'visible', timeout: 120 * 1000 });

  // await page.getByRole('textbox', { name: '▾ Controls' }).click();
  await page.getByRole('combobox', { name: 'models' }).selectOption('custom');
  await page.getByRole('textbox', { name: 'numRuns' }).click();
  await page.getByRole('textbox', { name: 'numRuns' }).fill('100');
  await page.getByRole('combobox', { name: 'backend' }).selectOption('tflite');
  await page.getByRole('combobox', { name: 'models' }).selectOption('custom');
  await page.getByPlaceholder('https://your-domain.com/model-path/model.json').click();
  await page.getByPlaceholder('https://your-domain.com/model-path/model.json').fill(url);

  let backendCheckbox = false;
  (backend.toLowerCase() === 'webnn') ? backendCheckbox = true : backendCheckbox = false;

  // Unchecked === Wasm SIMD
  // Checked === WebNN Delegate
  await page.getByRole('checkbox', { name: 'webnn delegate' }).setChecked(backendCheckbox);
  await page.getByRole('button', { name: 'Run benchmark' }).click();

  const result = page.locator('#timings tbody tr:nth-child(4)');
  await result.waitFor({ state: 'visible', timeout: 120 * 1000 });

  const average = await page.locator('#timings tbody tr:nth-child(4) td:nth-child(2)').allInnerTexts();
  console.log(`${backend}: ${average}`);
}

