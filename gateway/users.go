package main

// User accounts, self-serve API keys and prepaid credits: customers buy access to the API
// directly from us (separate from OpenRouter traffic, which uses provider-level keys).

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type User struct {
	ID        string    `json:"id"`
	Email     string    `json:"email"`
	PassHash  string    `json:"-"`
	Salt      string    `json:"-"`
	CreatedAt time.Time `json:"created_at"`
	CreditUSD float64   `json:"credit_usd"` // prepaid balance
	SpentUSD  float64   `json:"spent_usd"`  // lifetime
	IsAdmin   bool      `json:"is_admin"`
	Disabled  bool      `json:"disabled"`
}

type APIKey struct {
	Key       string    `json:"key"`    // full key, shown once on creation
	Prefix    string    `json:"prefix"` // display form: fk-abcd…
	Hash      string    `json:"-"`      // sha256 of the key (what we store/compare)
	UserID    string    `json:"user_id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	LastUsed  time.Time `json:"last_used,omitempty"`
	Revoked   bool      `json:"revoked"`
}

type Session struct {
	Token     string    `json:"token"`
	UserID    string    `json:"user_id"`
	ExpiresAt time.Time `json:"expires_at"`
}

func randHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func hashPass(pass, salt string) string {
	h := sha256.Sum256([]byte(salt + pass + "llmfast"))
	return hex.EncodeToString(h[:])
}

func hashKey(k string) string {
	h := sha256.Sum256([]byte(k))
	return hex.EncodeToString(h[:])
}

// ---------- store accessors ----------

func (s *Store) UserByEmail(email string) *User {
	email = strings.ToLower(strings.TrimSpace(email))
	for i := range s.Users {
		if strings.ToLower(s.Users[i].Email) == email {
			return &s.Users[i]
		}
	}
	return nil
}

func (s *Store) UserByID(id string) *User {
	for i := range s.Users {
		if s.Users[i].ID == id {
			return &s.Users[i]
		}
	}
	return nil
}

// CreateUser registers an account. The first account created becomes the admin.
func (s *Store) CreateUser(email, pass string) (*User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !strings.Contains(email, "@") {
		return nil, fmt.Errorf("valid email required")
	}
	if len(pass) < 8 {
		return nil, fmt.Errorf("password must be at least 8 characters")
	}
	if s.UserByEmail(email) != nil {
		return nil, fmt.Errorf("an account with that email already exists")
	}
	salt := randHex(8)
	u := User{ID: "usr-" + randHex(8), Email: strings.TrimSpace(email), Salt: salt, PassHash: hashPass(pass, salt),
		CreatedAt: time.Now(), IsAdmin: len(s.Users) == 0, CreditUSD: freeCreditUSD}
	s.Users = append(s.Users, u)
	s.save()
	return &s.Users[len(s.Users)-1], nil
}

func (s *Store) Login(email, pass string) (*Session, *User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.UserByEmail(email)
	if u == nil || u.PassHash != hashPass(pass, u.Salt) {
		return nil, nil, fmt.Errorf("wrong email or password")
	}
	if u.Disabled {
		return nil, nil, fmt.Errorf("account disabled")
	}
	sess := Session{Token: randHex(24), UserID: u.ID, ExpiresAt: time.Now().Add(30 * 24 * time.Hour)}
	s.Sessions = append(s.Sessions, sess)
	s.save()
	return &sess, u, nil
}

func (s *Store) UserBySession(token string) *User {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for _, sess := range s.Sessions {
		if sess.Token == token && sess.ExpiresAt.After(now) {
			return s.UserByID(sess.UserID)
		}
	}
	return nil
}

func (s *Store) Logout(token string) {
	s.mu.Lock()
	kept := s.Sessions[:0]
	for _, sess := range s.Sessions {
		if sess.Token != token {
			kept = append(kept, sess)
		}
	}
	s.Sessions = kept
	s.save()
	s.mu.Unlock()
}

func (s *Store) CreateAPIKey(userID, name string) APIKey {
	s.mu.Lock()
	defer s.mu.Unlock()
	raw := "fk-" + randHex(24)
	k := APIKey{Key: raw, Prefix: raw[:10] + "…", Hash: hashKey(raw), UserID: userID, Name: name, CreatedAt: time.Now()}
	stored := k
	stored.Key = "" // never persist the raw key
	s.APIKeys = append(s.APIKeys, stored)
	s.save()
	return k
}

func (s *Store) KeysOf(userID string) []APIKey {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []APIKey{}
	for _, k := range s.APIKeys {
		if k.UserID == userID && !k.Revoked {
			out = append(out, k)
		}
	}
	return out
}

func (s *Store) RevokeKey(userID, prefix string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.APIKeys {
		if s.APIKeys[i].UserID == userID && s.APIKeys[i].Prefix == prefix {
			s.APIKeys[i].Revoked = true
			s.save()
			return true
		}
	}
	return false
}

// AuthKey resolves an API key to its owner, enforcing revocation and credit balance.
// Legacy keys in Store.Keys (provider/OpenRouter keys) authenticate with no user attached.
func (s *Store) AuthKey(raw string) (userID string, ok bool, reason string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Keys[raw] {
		return "", true, ""
	}
	h := hashKey(raw)
	for i := range s.APIKeys {
		if s.APIKeys[i].Hash == h {
			if s.APIKeys[i].Revoked {
				return "", false, "api key revoked"
			}
			u := s.UserByID(s.APIKeys[i].UserID)
			if u == nil || u.Disabled {
				return "", false, "account disabled"
			}
			if u.CreditUSD <= 0 {
				return u.ID, false, "insufficient credit — top up to continue"
			}
			s.APIKeys[i].LastUsed = time.Now()
			return u.ID, true, ""
		}
	}
	return "", false, "invalid api key"
}

// Charge deducts a request's cost from the user's prepaid balance.
func (s *Store) Charge(userID string, usd float64) {
	if userID == "" || usd == 0 {
		return
	}
	s.mu.Lock()
	if u := s.UserByID(userID); u != nil {
		u.CreditUSD -= usd
		u.SpentUSD += usd
		s.save()
	}
	s.mu.Unlock()
}

func (s *Store) AddCredit(userID string, usd float64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.UserByID(userID)
	if u == nil {
		return fmt.Errorf("unknown user")
	}
	u.CreditUSD += usd
	s.save()
	return nil
}

// ---------- HTTP ----------

const freeCreditUSD = 1.0 // starter credit on signup

var sessionCookie = "llmfast_session"

func (s *Server) currentUser(r *http.Request) *User {
	if c, err := r.Cookie(sessionCookie); err == nil {
		if u := s.store.UserBySession(c.Value); u != nil {
			return u
		}
	}
	if t := bearer(r); t != "" {
		if u := s.store.UserBySession(t); u != nil {
			return u
		}
	}
	return nil
}

func (s *Server) handleSignup(w http.ResponseWriter, r *http.Request) {
	var in struct{ Email, Password string }
	json.NewDecoder(r.Body).Decode(&in)
	u, err := s.store.CreateUser(in.Email, in.Password)
	if err != nil {
		apiError(w, 400, err.Error())
		return
	}
	sess, _, err := s.store.Login(in.Email, in.Password)
	if err != nil {
		apiError(w, 500, err.Error())
		return
	}
	setSession(w, sess)
	writeJSON(w, 200, map[string]any{"user": u, "token": sess.Token})
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	var in struct{ Email, Password string }
	json.NewDecoder(r.Body).Decode(&in)
	sess, u, err := s.store.Login(in.Email, in.Password)
	if err != nil {
		apiError(w, 401, err.Error())
		return
	}
	setSession(w, sess)
	writeJSON(w, 200, map[string]any{"user": u, "token": sess.Token})
}

func setSession(w http.ResponseWriter, sess *Session) {
	http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: sess.Token, Path: "/", Expires: sess.ExpiresAt,
		HttpOnly: true, SameSite: http.SameSiteLaxMode})
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookie); err == nil {
		s.store.Logout(c.Value)
	}
	http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: "", Path: "/", MaxAge: -1})
	writeJSON(w, 200, map[string]string{"status": "ok"})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request) {
	u := s.currentUser(r)
	if u == nil {
		apiError(w, 401, "not signed in")
		return
	}
	writeJSON(w, 200, map[string]any{"user": u, "keys": s.store.KeysOf(u.ID), "usage": s.store.UserSummary(u.ID, 30*24*time.Hour)})
}

func (s *Server) handleUserKeys(w http.ResponseWriter, r *http.Request) {
	u := s.currentUser(r)
	if u == nil {
		apiError(w, 401, "not signed in")
		return
	}
	switch r.Method {
	case "POST":
		var in struct{ Name string }
		json.NewDecoder(r.Body).Decode(&in)
		if in.Name == "" {
			in.Name = "default"
		}
		writeJSON(w, 200, s.store.CreateAPIKey(u.ID, in.Name))
	case "DELETE":
		if !s.store.RevokeKey(u.ID, r.URL.Query().Get("prefix")) {
			apiError(w, 404, "key not found")
			return
		}
		writeJSON(w, 200, map[string]string{"status": "ok"})
	default:
		writeJSON(w, 200, map[string]any{"keys": s.store.KeysOf(u.ID)})
	}
}

// Top-up: credits are added by an admin (or a payment webhook once a processor is connected).
func (s *Server) handleTopup(w http.ResponseWriter, r *http.Request) {
	var in struct {
		UserID string  `json:"user_id"`
		Amount float64 `json:"amount_usd"`
	}
	json.NewDecoder(r.Body).Decode(&in)
	if in.Amount <= 0 {
		apiError(w, 400, "amount_usd must be positive")
		return
	}
	if err := s.store.AddCredit(in.UserID, in.Amount); err != nil {
		apiError(w, 404, err.Error())
		return
	}
	writeJSON(w, 200, map[string]string{"status": "ok"})
}

func (s *Server) handleAdminUsers(w http.ResponseWriter, r *http.Request) {
	s.store.mu.Lock()
	users := append([]User{}, s.store.Users...)
	s.store.mu.Unlock()
	out := []map[string]any{}
	for _, u := range users {
		sum := s.store.UserSummary(u.ID, 30*24*time.Hour)
		out = append(out, map[string]any{"user": u, "usage": sum})
	}
	writeJSON(w, 200, map[string]any{"users": out})
}

var _ = sync.Mutex{}
