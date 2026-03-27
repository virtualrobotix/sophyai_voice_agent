#!/bin/bash
# Rinnovo certificato Let's Encrypt per chatbotdev.sophyai.io
# Uso: sudo ./renew-cert.sh
#   oppure: sudo ./renew-cert.sh --force   (forza il rinnovo anche se non scaduto)

set -e

DOMAIN="chatbotdev.sophyai.io"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="$DIR/certs"
LE_LIVE="/etc/letsencrypt/live/$DOMAIN"

if [ "$EUID" -ne 0 ]; then
    echo "Errore: esegui con sudo"
    echo "  sudo $0"
    exit 1
fi

echo "=== Rinnovo certificato SSL per $DOMAIN ==="
echo ""

# Verifica certificato attuale
if [ -f "$CERTS_DIR/cert.pem" ]; then
    EXPIRY=$(openssl x509 -in "$CERTS_DIR/cert.pem" -noout -enddate 2>/dev/null | cut -d= -f2)
    ISSUER=$(openssl x509 -in "$CERTS_DIR/cert.pem" -noout -issuer 2>/dev/null)
    echo "Certificato attuale:"
    echo "  Scadenza: $EXPIRY"
    echo "  Emesso da: $ISSUER"
    echo ""
fi

# Rinnovo tramite certbot
if [ "$1" = "--force" ]; then
    echo "Rinnovo forzato..."
    certbot renew --cert-name "$DOMAIN" --force-renewal
else
    echo "Rinnovo (solo se necessario)..."
    certbot renew --cert-name "$DOMAIN"
fi

# Copia i certificati aggiornati nel progetto
if [ -f "$LE_LIVE/fullchain.pem" ] && [ -f "$LE_LIVE/privkey.pem" ]; then
    echo ""
    echo "Copio certificati in $CERTS_DIR ..."
    cp "$LE_LIVE/fullchain.pem" "$CERTS_DIR/cert.pem"
    cp "$LE_LIVE/privkey.pem"   "$CERTS_DIR/key.pem"

    OWNER=$(stat -c '%U:%G' "$DIR" 2>/dev/null || echo "laserlab:laserlab")
    chown "$OWNER" "$CERTS_DIR/cert.pem" "$CERTS_DIR/key.pem"
    chmod 644 "$CERTS_DIR/cert.pem"
    chmod 600 "$CERTS_DIR/key.pem"

    echo "Riavvio container..."
    docker restart voice-agent-web livekit-tls-proxy 2>/dev/null || \
        docker compose restart web livekit-tls-proxy 2>/dev/null || \
        echo "ATTENZIONE: riavvio container fallito, fallo manualmente"

    sleep 3

    # Verifica
    NEW_EXPIRY=$(openssl x509 -in "$CERTS_DIR/cert.pem" -noout -enddate 2>/dev/null | cut -d= -f2)
    NEW_ISSUER=$(openssl x509 -in "$CERTS_DIR/cert.pem" -noout -issuer 2>/dev/null)
    echo ""
    echo "Nuovo certificato:"
    echo "  Scadenza: $NEW_EXPIRY"
    echo "  Emesso da: $NEW_ISSUER"

    # Test HTTPS
    if curl -sf https://localhost:8443/api/health > /dev/null 2>&1; then
        echo ""
        echo "HTTPS attivo e funzionante"
    else
        echo ""
        echo "ATTENZIONE: HTTPS non risponde, controlla i log con: docker logs voice-agent-web"
    fi
else
    echo ""
    echo "ERRORE: certificati Let's Encrypt non trovati in $LE_LIVE"
    echo "Esegui prima: certbot certonly --manual --preferred-challenges dns -d $DOMAIN"
    exit 1
fi

echo ""
echo "=== Rinnovo completato ==="
