#!/bin/bash
# =============================================================================
# Script di inizializzazione SIP per LiveKit
# Configura trunk e dispatch rules che non sono persistenti in LiveKit
# Questo script deve essere eseguito dopo l'avvio di LiveKit
# =============================================================================

set -e

# Configurazione
LIVEKIT_URL="${LIVEKIT_URL:-http://localhost:7880}"
LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret_dev_key_change_in_production}"
SIP_TRUNK_NAME="${SIP_TRUNK_NAME:-aims-dev-trunk}"
SIP_INBOUND_NUMBER="${SIP_INBOUND_NUMBER:-+3901119517860}"
SIP_ALLOWED_ADDRESS="${SIP_ALLOWED_ADDRESS:-aims-dev-trunk.pstn.frankfurt.twilio.com}"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# #region agent log
DEBUG_LOG_PATH="${DEBUG_LOG_PATH:-/home/laserlab/lavoro/Progetti_2026/sophyai_voice_agent/.cursor/debug-fac0c1.log}"
debug_log() {
    local hypothesis="$1"
    local location="$2"
    local message="$3"
    local data_json="$4"
    python3 - <<PY || true
import json, time, os
_raw = """$data_json"""
try:
    _data = json.loads(_raw)
except Exception:
    _data = {"raw": _raw[:400]}
line = {
  "sessionId":"fac0c1",
  "hypothesisId":"$hypothesis",
  "location":"$location",
  "message":"$message",
  "data":_data,
  "timestamp":int(time.time()*1000),
  "runId":"sip-init"
}
try:
    os.makedirs(os.path.dirname("$DEBUG_LOG_PATH"), exist_ok=True)
    with open("$DEBUG_LOG_PATH","a",encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\\n")
except Exception:
    pass
PY
}
# #endregion

# Genera JWT token per API LiveKit
generate_token() {
    python3 -c "
import jwt, time
token = jwt.encode({
    'iss': '${LIVEKIT_API_KEY}',
    'sub': 'sip-init',
    'iat': int(time.time()),
    'exp': int(time.time()) + 3600,
    'video': {'roomAdmin': True, 'room': '*'},
    'sip': {'admin': True, 'call': True}
}, '${LIVEKIT_API_SECRET}', algorithm='HS256')
print(token)
"
}

# Attendi che LiveKit sia pronto
wait_for_livekit() {
    log_info "Attendo LiveKit..."
    # #region agent log
    debug_log "H1-livekit" "init-sip.sh:wait_for_livekit" "wait_start" "{\"livekit_url\":\"${LIVEKIT_URL}\"}"
    # #endregion
    for i in {1..30}; do
        if curl -s "${LIVEKIT_URL}" > /dev/null 2>&1; then
            log_info "LiveKit pronto!"
            # #region agent log
            debug_log "H1-livekit" "init-sip.sh:wait_for_livekit" "wait_ready" "{\"attempt\":${i}}"
            # #endregion
            return 0
        fi
        sleep 2
    done
    log_error "LiveKit non risponde dopo 60 secondi"
    # #region agent log
    debug_log "H1-livekit" "init-sip.sh:wait_for_livekit" "wait_timeout" "{}"
    # #endregion
    return 1
}

# Crea SIP Inbound Trunk per Twilio
create_inbound_trunk() {
    local TOKEN=$(generate_token)
    
    log_info "Creazione SIP Inbound Trunk (${SIP_TRUNK_NAME})..." >&2
    
    RESPONSE=$(curl -s -X POST "${LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPInboundTrunk" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d "{
            \"trunk\": {
                \"name\": \"${SIP_TRUNK_NAME}\",
                \"numbers\": [\"${SIP_INBOUND_NUMBER}\"],
                \"allowed_addresses\": [
                    \"${SIP_ALLOWED_ADDRESS}\",
                    \"35.156.191.128/25\",
                    \"35.156.191.0/25\",
                    \"54.171.127.192/26\",
                    \"54.172.60.0/23\",
                    \"54.244.51.0/24\"
                ],
                \"metadata\": \"{\\\"provider\\\": \\\"twilio\\\", \\\"trunk_type\\\": \\\"aims\\\"}\"
            }
        }")
    # #region agent log
    if echo "$RESPONSE" | python3 -c "import sys,json; json.load(sys.stdin)" >/dev/null 2>&1; then
        debug_log "H2-trunk" "init-sip.sh:create_inbound_trunk" "create_trunk_response" "$RESPONSE"
    else
        debug_log "H2-trunk" "init-sip.sh:create_inbound_trunk" "create_trunk_non_json" "{\"raw\":$(python3 - <<'PY'
import json,sys
print(json.dumps(sys.stdin.read()[:300]))
PY
<<<"$RESPONSE")}"
    fi
    # #endregion
    
    TRUNK_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sip_trunk_id') or d.get('sipTrunkId') or '')" 2>/dev/null || echo "")
    if [ -n "$TRUNK_ID" ]; then
        log_info "Trunk creato: ${TRUNK_ID}" >&2
        echo "$TRUNK_ID"
    else
        log_warn "Trunk potrebbe già esistere o errore: $RESPONSE" >&2
        echo ""
    fi
}

# Crea Dispatch Rule per instradare chiamate all'agent
create_dispatch_rule() {
    local TOKEN=$(generate_token)
    local TRUNK_IDS="$1"
    
    log_info "Creazione Dispatch Rule..."
    
    # Dispatch rule che crea una room per ogni chiamata (formato LiveKit corretto)
    RESPONSE=$(curl -s -X POST "${LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPDispatchRule" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d "{
            \"rule\": {
                \"name\": \"default-inbound\",
                \"dispatchRuleIndividual\": {
                    \"roomPrefix\": \"sip-call-\"
                }
            },
            \"trunk_ids\": [\"${TRUNK_IDS}\"]
        }")
    
    if echo "$RESPONSE" | grep -q "sip_dispatch_rule_id"; then
        RULE_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sip_dispatch_rule_id',''))" 2>/dev/null)
        log_info "Dispatch Rule creata: ${RULE_ID}"
    else
        log_warn "Dispatch Rule potrebbe già esistere o errore: $RESPONSE"
    fi
}

# Verifica configurazione esistente
check_existing_config() {
    local TOKEN=$(generate_token)
    
    # Check trunks
    TRUNKS=$(curl -s -X POST "${LIVEKIT_URL}/twirp/livekit.SIP/ListSIPInboundTrunk" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d '{}')
    
    TRUNK_COUNT=$(echo "$TRUNKS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('items',[])))" 2>/dev/null || echo "0")
    
    # Check dispatch rules
    RULES=$(curl -s -X POST "${LIVEKIT_URL}/twirp/livekit.SIP/ListSIPDispatchRule" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d '{}')
    
    RULE_COUNT=$(echo "$RULES" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('items',[])))" 2>/dev/null || echo "0")
    # #region agent log
    debug_log "H3-routing" "init-sip.sh:check_existing_config" "current_config_counts" "{\"trunk_count\":${TRUNK_COUNT},\"rule_count\":${RULE_COUNT}}"
    # #endregion
    
    log_info "Configurazione attuale: ${TRUNK_COUNT} trunk, ${RULE_COUNT} dispatch rules"
    
    if [ "$TRUNK_COUNT" -gt 0 ] && [ "$RULE_COUNT" -gt 0 ]; then
        return 0  # Già configurato
    fi
    return 1  # Necessita configurazione
}

# Main
main() {
    log_info "=== Inizializzazione SIP LiveKit ==="
    
    wait_for_livekit || exit 1
    
    if check_existing_config; then
        log_info "SIP già configurato, skip"
        exit 0
    fi
    
    log_info "Configurazione SIP mancante, creo trunk e rules..."
    
    TRUNK_ID=$(create_inbound_trunk | tail -n1)
    create_dispatch_rule "$TRUNK_ID"
    
    log_info "=== Configurazione SIP completata ==="
    
    # Verifica finale
    check_existing_config
}

main "$@"
