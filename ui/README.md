# OCRlty UI

Минимальный P0 frontend для просмотра runs и extract artifacts.

## Запуск

```bash
cd ui
npm install
npm run dev
```

UI будет доступен на `http://localhost:5173`.

## Логин

На странице `/login` укажите:
- API Base URL (по умолчанию: `http://127.0.0.1:8080`)
- API Key

Кнопка `Log in` проверяет ключ через `GET /v1/me`.

## LocalStorage keys

Приложение использует ключи:
- `ocrlty_api_base_url`
- `ocrlty_api_key`
