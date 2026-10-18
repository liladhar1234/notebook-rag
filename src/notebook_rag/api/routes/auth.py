"""POST /auth/register, POST /auth/login, POST /auth/logout, GET /auth/me.

The session is an opaque random token in an HttpOnly cookie (JavaScript cannot read it).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from notebook_rag.api.deps import COOKIE_NAME, get_current_user, get_usage_repository, get_user_repository
from notebook_rag.api.security import (
    DUMMY_HASH,
    LoginLimiter,
    hash_password,
    new_token,
    normalize_email,
    valid_email,
    verify_password,
)
from notebook_rag.config import get_settings
from notebook_rag.registry.usage import UsageRepository
from notebook_rag.registry.users import EmailTaken, UserRecord, UserRepository

router = APIRouter()


class Credentials(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=128)


class UserInfo(BaseModel):
    user_id: str
    email: str


def _info(u: UserRecord) -> UserInfo:
    return UserInfo(user_id=u.user_id, email=u.email)


def _limiter(request: Request) -> LoginLimiter:
    if not hasattr(request.app.state, "login_limiter"):
        request.app.state.login_limiter = LoginLimiter()
    return request.app.state.login_limiter


async def _start_session(response: Response, users: UserRepository, user: UserRecord) -> None:
    cfg = get_settings()
    token = new_token()
    await users.create_session(user.user_id, token, cfg.session_days)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=cfg.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=cfg.cookie_secure,
        path="/",
    )


@router.post("/auth/register", response_model=UserInfo, status_code=201)
async def register(
    body: Credentials,
    response: Response,
    users: UserRepository = Depends(get_user_repository),
):
    cfg = get_settings()
    if not cfg.allow_signup:
        raise HTTPException(status_code=403, detail="Sign-up is disabled on this server.")
    email = normalize_email(body.email)
    if not valid_email(email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(body.password) < cfg.min_password_length:
        raise HTTPException(
            status_code=422, detail=f"Password must be at least {cfg.min_password_length} characters."
        )
    pw_hash = await asyncio.to_thread(hash_password, body.password)
    try:
        user = await users.create(email, pw_hash)
    except EmailTaken:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from None
    await _start_session(response, users, user)
    return _info(user)


@router.post("/auth/login", response_model=UserInfo)
async def login(
    body: Credentials,
    request: Request,
    response: Response,
    users: UserRepository = Depends(get_user_repository),
):
    email = normalize_email(body.email)
    key = f"{email}|{request.client.host if request.client else '-'}"
    limiter = _limiter(request)
    if limiter.blocked(key):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")

    user = await users.by_email(email)
    ok = await asyncio.to_thread(verify_password, body.password, user.password_hash if user else DUMMY_HASH)
    if not user or not ok:
        limiter.record_failure(key)
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    limiter.reset(key)
    await _start_session(response, users, user)
    return _info(user)


@router.post("/auth/logout", status_code=204)
async def logout(
    request: Request,
    response: Response,
    users: UserRepository = Depends(get_user_repository),
):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        await users.delete_session(token)
    response.delete_cookie(COOKIE_NAME, path="/")


@router.get("/auth/me", response_model=UserInfo)
async def me(user: UserRecord = Depends(get_current_user)):
    return _info(user)


class UsageInfo(BaseModel):
    used: int
    limit: int                 # per-user daily limit, 0 = unlimited
    remaining: int | None      # None when unlimited


@router.get("/auth/usage", response_model=UsageInfo)
async def my_usage(
    user: UserRecord = Depends(get_current_user),
    usage: UsageRepository = Depends(get_usage_repository),
):
    """How many of today's questions this account has used."""
    limit = get_settings().daily_questions_per_user
    used = await usage.used_today(user.user_id)
    return UsageInfo(used=used, limit=limit, remaining=max(limit - used, 0) if limit else None)
