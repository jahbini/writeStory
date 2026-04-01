# meta/sqlite.coffee
path = require 'path'
{ DatabaseSync } = require 'node:sqlite'

module.exports = (M, opts={}) ->
    baseDir = opts.baseDir ? process.cwd()
    dbFile = opts.sqliteFile ? 'runtime.sqlite'
    dbPath = if path.isAbsolute(dbFile) then dbFile else path.join(baseDir, dbFile)
    db = new DatabaseSync(dbPath)

    debugEnabled = ->
        try
            globalParams = M.theLowdown("params/_global.yaml")?.value ? {}
            globalParams.debug_sql is true
        catch then false

    debugLog = (parts...) ->
        return unless debugEnabled()
        console.log "[#{new Date().toISOString()}] [SQL]", parts...

    # HEY JIM! There is no repo-defined SQLite schema yet, so this file
    # creates the minimum schema needed for the request-key contract.
    db.exec """
    CREATE TABLE IF NOT EXISTS stories (
      story_id TEXT PRIMARY KEY,
      title TEXT,
      text TEXT
    );

    CREATE TABLE IF NOT EXISTS story_parts (
      story_id TEXT PRIMARY KEY,
      scene TEXT,
      arrival TEXT,
      disturbance TEXT,
      reflection TEXT,
      realization TEXT
    );

    CREATE TABLE IF NOT EXISTS kag_entries (
      story_id TEXT NOT NULL,
      entry_index INTEGER NOT NULL,
      doc_id TEXT,
      paragraph_index TEXT,
      keyword TEXT,
      headline TEXT,
      entry_json TEXT,
      PRIMARY KEY (story_id, entry_index)
    );

    CREATE INDEX IF NOT EXISTS idx_kag_entries_story_id
      ON kag_entries (story_id);

    CREATE INDEX IF NOT EXISTS idx_kag_entries_keyword
      ON kag_entries (keyword);
    """

    FORMATTERS =
      json:
        read: (value) ->
          throw new Error "sqlite meta expected object for .json" unless value? and typeof value is 'object' and not Array.isArray(value)
          value
        write: (value) ->
          throw new Error "sqlite meta expected object for .json" unless value? and typeof value is 'object' and not Array.isArray(value)
          value

      jsonl:
        read: (value) ->
          throw new Error "sqlite meta expected array for .jsonl" unless Array.isArray(value)
          value
        write: (value) ->
          throw new Error "sqlite meta expected array for .jsonl" unless Array.isArray(value)
          value

      txt:
        read: (value) ->
          if typeof value is 'string'
            value
          else
            # HEY JIM! The directive requires txt support but does not define
            # a canonical text projection for structured values.
            JSON.stringify(value, null, 2)
        write: (value) ->
          if typeof value is 'string'
            value
          else
            JSON.stringify(value, null, 2)

      csv:
        read: (value) ->
          if Array.isArray(value)
            value
          else if value? and typeof value is 'object'
            value
          else
            throw new Error "sqlite meta expected object or array for .csv"
        write: (value) ->
          if Array.isArray(value)
            value
          else if value? and typeof value is 'object'
            value
          else
            throw new Error "sqlite meta expected object or array for .csv"

    REQUESTS = [
      {
        name: 'storyByID'
        regex: /^storyByID\{([^}]+)\}$/
        allowedSuffixes: ['json', 'txt', 'csv']
        read: (db, storyID) ->
          row = db.prepare("""
            SELECT story_id, title, text
            FROM stories
            WHERE story_id = ?
          """).get(storyID)

          throw new Error "sqlite meta missing storyByID #{storyID}" unless row?

          {
            story_id: row.story_id
            title: row.title
            text: row.text
          }
        write: (db, value, storyID) ->
          throw new Error "sqlite meta storyByID write expects object" unless value? and typeof value is 'object' and not Array.isArray(value)
          writeStoryID = value.story_id ? storyID
          throw new Error "sqlite meta storyByID story_id mismatch" unless writeStoryID is storyID

          db.prepare("""
            INSERT INTO stories (story_id, title, text)
            VALUES (?, ?, ?)
            ON CONFLICT(story_id) DO UPDATE SET
              title = excluded.title,
              text = excluded.text
          """).run(
            writeStoryID
            value.title ? null
            value.text ? null
          )

          {
            story_id: writeStoryID
            title: value.title ? null
            text: value.text ? null
          }
      }

      {
        name: 'partsFor'
        regex: /^partsFor\{([^}]+)\}$/
        allowedSuffixes: ['json', 'txt', 'csv']
        read: (db, storyID) ->
          row = db.prepare("""
            SELECT story_id, scene, arrival, disturbance, reflection, realization
            FROM story_parts
            WHERE story_id = ?
          """).get(storyID)

          throw new Error "sqlite meta missing partsFor #{storyID}" unless row?

          {
            story_id: row.story_id
            parts:
              scene: row.scene
              arrival: row.arrival
              disturbance: row.disturbance
              reflection: row.reflection
              realization: row.realization
          }
        write: (db, value, storyID) ->
          throw new Error "sqlite meta partsFor write expects object" unless value? and typeof value is 'object' and not Array.isArray(value)
          throw new Error "sqlite meta partsFor write expects value.parts" unless value.parts? and typeof value.parts is 'object' and not Array.isArray(value.parts)
          writeStoryID = value.story_id ? storyID
          throw new Error "sqlite meta partsFor story_id mismatch" unless writeStoryID is storyID

          db.prepare("""
            INSERT INTO story_parts (
              story_id, scene, arrival, disturbance, reflection, realization
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(story_id) DO UPDATE SET
              scene = excluded.scene,
              arrival = excluded.arrival,
              disturbance = excluded.disturbance,
              reflection = excluded.reflection,
              realization = excluded.realization
          """).run(
            writeStoryID
            value.parts.scene ? null
            value.parts.arrival ? null
            value.parts.disturbance ? null
            value.parts.reflection ? null
            value.parts.realization ? null
          )

          {
            story_id: writeStoryID
            parts:
              scene: value.parts.scene ? null
              arrival: value.parts.arrival ? null
              disturbance: value.parts.disturbance ? null
              reflection: value.parts.reflection ? null
              realization: value.parts.realization ? null
          }
      }

      {
        name: 'kagFor'
        regex: /^kagFor\{([^}]+)\}$/
        allowedSuffixes: ['json', 'txt', 'csv']
        read: (db, storyID) ->
          rows = db.prepare("""
            SELECT story_id, entry_index, doc_id, paragraph_index, keyword, headline, entry_json
            FROM kag_entries
            WHERE story_id = ?
            ORDER BY entry_index ASC
          """).all(storyID)

          throw new Error "sqlite meta missing kagFor #{storyID}" unless rows.length

          entries = []
          seenKeywords = new Set()
          keywords = []

          for row in rows
            entry = null
            if row.entry_json?
              entry = JSON.parse(row.entry_json)
            else
              entry =
                story_id: row.story_id
                entry_index: row.entry_index
                doc_id: row.doc_id
                paragraph_index: row.paragraph_index
                keyword: row.keyword
                headline: row.headline

            entries.push entry

            keyword = row.keyword
            if keyword? and not seenKeywords.has(keyword)
              seenKeywords.add keyword
              keywords.push keyword

          {
            story_id: storyID
            entries: entries
            keywords: keywords
          }
        write: (db, value, storyID) ->
          throw new Error "sqlite meta kagFor write expects object" unless value? and typeof value is 'object' and not Array.isArray(value)
          throw new Error "sqlite meta kagFor write expects entries array" unless Array.isArray(value.entries)
          throw new Error "sqlite meta kagFor write expects keywords array" unless Array.isArray(value.keywords)
          writeStoryID = value.story_id ? storyID
          throw new Error "sqlite meta kagFor story_id mismatch" unless writeStoryID is storyID

          # HEY JIM! Queryable KAG keywords are stored per entry row. Keywords
          # present only in value.keywords but not attached to an entry are not
          # independently queryable because the prompt did not define a separate
          # storage shape for them.
          db.exec 'BEGIN'
          try
            db.prepare("""
              DELETE FROM kag_entries
              WHERE story_id = ?
            """).run(writeStoryID)

            insertStatement = db.prepare("""
              INSERT INTO kag_entries (
                story_id, entry_index, doc_id, paragraph_index, keyword, headline, entry_json
              )
              VALUES (?, ?, ?, ?, ?, ?, ?)
            """)

            entryIndex = 1
            for entry in value.entries
              docID = entry?.doc_id ? entry?.meta?.doc_id ? null
              paragraphIndex = entry?.paragraph_index ? entry?.meta?.paragraph_index ? null
              keyword = entry?.keyword ? null
              headline = entry?.headline ? entry?.label ? entry?.text ? null

              insertStatement.run(
                writeStoryID
                entryIndex
                docID
                paragraphIndex
                keyword
                headline
                JSON.stringify(entry)
              )

              entryIndex += 1

            db.exec 'COMMIT'
          catch err
            try db.exec 'ROLLBACK' catch then null
            throw err

          {
            story_id: writeStoryID
            entries: value.entries
            keywords: value.keywords
          }
      }

      {
        name: 'storiesWithKag'
        regex: /^storiesWithKag\{([^}]+)\}$/
        allowedSuffixes: ['jsonl', 'txt', 'csv']
        read: (db, keyword) ->
          db.prepare("""
            SELECT DISTINCT stories.story_id, stories.title, stories.text
            FROM stories
            INNER JOIN kag_entries
              ON kag_entries.story_id = stories.story_id
            WHERE kag_entries.keyword = ?
            ORDER BY stories.story_id ASC
          """).all(keyword)
        write: null
      }

      {
        name: 'storiesMissingKag'
        regex: /^storiesMissingKag$/
        allowedSuffixes: ['jsonl', 'txt', 'csv']
        read: (db) ->
          db.prepare("""
            SELECT stories.story_id, stories.title, stories.text
            FROM stories
            WHERE NOT EXISTS (
              SELECT 1
              FROM kag_entries
              WHERE kag_entries.story_id = stories.story_id
            )
            ORDER BY stories.story_id ASC
          """).all()
        write: null
      }
    ]

    M.addMetaRule "sqlite",
      /\.(json|jsonl|txt|csv)$/i,
      (key, value) ->
        debugLog "meta key", key, "write?", value isnt undefined

        suffixMatch = key.match /\.([A-Za-z0-9]+)$/
        return undefined unless suffixMatch?

        suffix = suffixMatch[1].toLowerCase()
        formatter = FORMATTERS[suffix]
        return undefined unless formatter?

        requestKey = key.replace /\.[A-Za-z0-9]+$/, ''
        debugLog "parsed", "requestKey=#{requestKey}", "suffix=#{suffix}"

        matchedRequest = null
        matchedArgs = null

        for request in REQUESTS
          match = requestKey.match request.regex
          debugLog "regex check", request.name, String(request.regex), "matched=#{!!match}"
          continue unless match?
          matchedRequest = request
          matchedArgs = match.slice(1)
          break

        unless matchedRequest?
          debugLog "no sqlite request match", requestKey
          return undefined

        unless matchedRequest.allowedSuffixes.includes suffix
          throw new Error "sqlite meta request #{matchedRequest.name} does not allow .#{suffix}"

        debugLog "matched request", matchedRequest.name, "args=#{JSON.stringify(matchedArgs)}"

        if value is undefined
          debugLog "read start", matchedRequest.name, "args=#{JSON.stringify(matchedArgs)}"
          result = matchedRequest.read(db, matchedArgs...)
          if Array.isArray(result)
            debugLog "read result", matchedRequest.name, "rows=#{result.length}"
          else
            debugLog "read result", matchedRequest.name, "type=#{typeof result}"
          return formatter.read(result)

        throw new Error "sqlite meta request #{matchedRequest.name} is read-only" unless typeof matchedRequest.write is 'function'

        decoded = formatter.write(value)
        if Array.isArray(decoded)
          debugLog "write start", matchedRequest.name, "args=#{JSON.stringify(matchedArgs)}", "rows=#{decoded.length}"
        else
          debugLog "write start", matchedRequest.name, "args=#{JSON.stringify(matchedArgs)}", "type=#{typeof decoded}"
        matchedRequest.write(db, decoded, matchedArgs...)
        debugLog "write done", matchedRequest.name, "args=#{JSON.stringify(matchedArgs)}"
