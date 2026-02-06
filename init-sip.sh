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

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

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
    for i in {1..30}; do
        if curl -s "${LIVEKIT_URL}" > /dev/null 2>&1; then
            log_info "LiveKit pronto!"
            return 0
        fi
        sleep 2
    done
    log_error "LiveKit non risponde dopo 60 secondi"
    return 1
}

# Crea SIP Inbound Trunk per Twilio
create_inbound_trunk() {
    local TOKEN=$(generate_token)
    
    log_info "Creazione SIP Inbound Trunk per Twilio..."
    
    RESPONSE=$(curl -s -X POST "${LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPInboundTrunk" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d '{
            "trunk": {
                "name": "Twilio AIMS Trunk",
                "numbers": ["+3901119517814"],
                "allowed_addresses": [
                    "aims-dev-trunk.pstn.twilio.com",
                    "54.172.60.0/23",
                    "54.244.51.0/24",
                    "35.156.191.128/25",
                    "54.171.127.192/26",
                    "35.156.191.0/25"
                ],
                "metadata": "{\"provider\": \"twilio\", \"trunk_type\": \"aims\"}"
            }
        }')
    
    if echo "$RESPONSE" | grep -q "sipTrunkId"; then
        TRUNK_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sipTrunkId',''))" 2>/dev/null)
        log_info "Trunk creato: ${TRUNK_ID}"
        echo "$TRUNK_ID"
    else
        log_warn "Trunk potrebbe già esistere o errore: $RESPONSE"
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
        -d '{
            "rule": {
                "name": "Default Inbound Handler",
                "dispatchRuleIndividual": {
                    "roomPrefix": "sip-call-"
                }
            }
        }')
    
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
    
    TRUNK_ID=$(create_inbound_trunk)
    create_dispatch_rule "$TRUNK_ID"
    
    log_info "=== Configurazione SIP completata ==="
    
    # Verifica finale
    check_existing_config
}

main "$@"
