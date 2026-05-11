# phi15_tokenizer.coffee
# Phi-1.5 GPT-2 style tokenizer, CoffeeScript port
# Uses tokenizer_data.json with { tokenToId, idToToken, merges }

fs   = require 'fs'
path = require 'path'

class Phi15Tokenizer

  constructor: (opts = {}) ->
    baseDir  = opts.baseDir ? __dirname
    jsonPath = path.join(baseDir, 'tokenizer_data.json')

    raw = fs.readFileSync jsonPath, 'utf8'
    data = JSON.parse raw

    @tokenToId = data.tokenToId   # { tokenString -> id }
    @idToToken = data.idToToken   # [ tokenString ]
    merges     = data.merges      # [ [s1, s2], ... ]

    # --- Build BPE ranks (pair -> rank) ---
    @bpeRanks = {}
    for pair, idx in merges
      # encode pair as a single key; \u0000 is safe separator
      key = pair[0] + '\u0000' + pair[1]
      @bpeRanks[key] = idx

    # --- Build byte encoder/decoder (GPT-2 bytes_to_unicode) ---
    bs = []
    cs = []

    i = 33
    while i < 127
      bs.push i
      cs.push i
      i++

    i = 161
    while i < 173
      bs.push i
      cs.push i
      i++

    i = 174
    while i < 256
      bs.push i
      cs.push i
      i++

    n = 0
    for b in [0..255]
      unless b in bs
        bs.push b
        cs.push 256 + n
        n++

    @byteEncoder = {}
    @byteDecoder = {}

    for b, idx in bs
      codePoint = cs[idx]
      ch = String.fromCharCode codePoint
      @byteEncoder[b] = ch
      @byteDecoder[ch] = b

    # --- GPT-2 tokenization regex pattern ---
    # 's, 't, 're, 've, 'm, 'll, 'd, words, numbers, punctuation, and whitespace
    @tokenPattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

    # BPE cache for speed
    @bpeCache = {}

  # --- Helper: given an array of symbols, return list of adjacent pairs ---
  getPairs: (word) ->
    pairs = []
    if word.length > 1
      prev = word[0]
      for i in [1...word.length]
        curr = word[i]
        pairs.push [prev, curr]
        prev = curr
    pairs

  # --- Core BPE merge loop on a single “byte-encoded” token string ---
  bpe: (token) ->
    cached = @bpeCache[token]
    return cached if cached?

    word = (c for c in token.split '')
    if word.length <= 1
      @bpeCache[token] = token
      return token

    pairs = @getPairs word
    while true
      minRank = Infinity
      bigram  = null

      for p in pairs
        key  = p[0] + '\u0000' + p[1]
        rank = @bpeRanks[key]
        if rank? and rank < minRank
          minRank = rank
          bigram  = p

      break unless bigram?

      first  = bigram[0]
      second = bigram[1]

      newWord = []
      i = 0
      while i < word.length
        if i < word.length - 1 and word[i] is first and word[i + 1] is second
          newWord.push first + second
          i += 2
        else
          newWord.push word[i]
          i++

      word = newWord
      break if word.length is 1
      pairs = @getPairs word

    result = word.join ' '
    @bpeCache[token] = result
    result

  # --- Encode text -> array of token IDs ---
  encode: (text) ->
    tokens = []
    pattern = @tokenPattern
    pattern.lastIndex = 0

    while true
      m = pattern.exec text
      break unless m?
      token = m[0]

      # Convert UTF-8 bytes to GPT-2 “byte-unicode” string
      buf = Buffer.from token, 'utf8'
      chars = []
      for byte in buf
        mapped = @byteEncoder[byte]
        unless mapped?
          throw new Error "No byteEncoder mapping for byte #{byte}"
        chars.push mapped
      tokenBytes = chars.join ''

      # Apply BPE
      bpeOut = @bpe tokenBytes
      parts  = bpeOut.split ' '

      for part in parts
        id = @tokenToId[part]
        unless id?
          throw new Error "Unknown BPE token '#{part}' (no entry in tokenToId)"
        tokens.push id

    tokens

  # --- Decode array of token IDs -> text ---
  decode: (ids) ->
    outTokens = []
    for id in ids
      token = @idToToken[id]
      unless token?
        throw new Error "Unknown token id #{id}"
      outTokens.push token

    byteChars = outTokens.join ''
    bytes = []
    for ch in byteChars.split ''
      byte = @byteDecoder[ch]
      unless byte?
        throw new Error "No byteDecoder mapping for char '#{ch}' (code #{ch.charCodeAt 0})"
      bytes.push byte

    Buffer.from(bytes).toString 'utf8'


# --- Module exports ---
module.exports =
    new Phi15Tokenizer()


# --- Optional CLI test ---
if require.main is module
  tokenizer = new Phi15Tokenizer()
  prompt = process.argv.slice(2).join(' ') or 'one two three four'
  ids = tokenizer.encode prompt
  console.log "Prompt:", prompt
  console.log "Token IDs:", JSON.stringify ids
  console.log "Decoded:", tokenizer.decode ids
