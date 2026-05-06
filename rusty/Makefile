verify:
	node verify_bridge.mjs

test-meta:
	@if [ ! -x "../node_modules/.bin/coffee" ]; then echo "CoffeeScript runner missing: ../node_modules/.bin/coffee"; exit 2; fi
	../node_modules/.bin/coffee test_meta_bridge.coffee
