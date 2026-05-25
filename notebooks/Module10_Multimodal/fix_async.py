"""Fix asyncio.run() calls to work with nbconvert's event loop."""
import json, sys

nb = json.load(open('03_video_understanding.ipynb'))

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'asyncio.run(main())' in src:
            src = src.replace('asyncio.run(main())',
                'loop = asyncio.get_event_loop_policy().get_event_loop()\n'
                'loop.run_until_complete(main())')
            cell['source'] = src
            print('Fixed Micro Practice 3 cell')
        if 'asyncio.run(demo())' in src:
            src = src.replace('asyncio.run(demo())',
                'loop = asyncio.get_event_loop_policy().get_event_loop()\n'
                'loop.run_until_complete(demo())')
            cell['source'] = src
            print('Fixed StreamEngine demo cell')

json.dump(nb, open('03_video_understanding.ipynb', 'w'), indent=1, ensure_ascii=False)
print('Saved.')
