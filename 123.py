-- подключиться к базе nwc (если выполняете из psql)
\c nwc

-- 1. создать роль (пользователя) без прав на создание объектов
CREATE ROLE nwc_reader
LOGIN
PASSWORD 'STRONG_PASSWORD_HERE'
NOSUPERUSER
NOCREATEDB
NOCREATEROLE
NOINHERIT;

-- 2. разрешить подключение к базе
GRANT CONNECT ON DATABASE nwc TO nwc_reader;

-- 3. доступ к схеме
GRANT USAGE ON SCHEMA public TO nwc_reader;

-- 4. доступ только на чтение ко всем существующим таблицам
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nwc_reader;

-- 5. доступ к последовательностям (если есть serial / identity)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nwc_reader;

-- 6. права по умолчанию для будущих таблиц
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO nwc_reader;

-- 7. права по умолчанию для будущих последовательностей
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO nwc_reader;
