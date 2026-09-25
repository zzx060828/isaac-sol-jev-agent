"""Small synthetic descriptions for tests, not copied from an EID installation."""
from unittest.mock import patch

def descriptions(kind='collectible'):
    ids={'card':[4,5,6,9,10,20], 'pill':[0,4,5], 'horse_pill':[0,4,5]}.get(kind, [])
    return {i:dict(name=f'Test {kind} {i}',description=f'Synthetic known effect {i}',description_source='unit-test fixture') for i in ids}

def enable(test):
    for target in ['isaac_agent.knowledge.item_descriptions','test_consumable_decisions.item_descriptions']:
        context=patch(target,side_effect=descriptions)
        context.start();test.addCleanup(context.stop)
