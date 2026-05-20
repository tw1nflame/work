import asyncio
import os
import asyncpg


async def main():
    conn = await asyncpg.connect(
        os.environ["DATABASE_URL"],
        timeout=5,
        command_timeout=5,
    )

    value = await conn.fetchval("select 1")
    print(value)

    await conn.close()


asyncio.run(main())
