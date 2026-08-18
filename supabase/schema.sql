-- Setter frontend schema: profiles, saved routes, ratings.
-- Run this once in the Supabase project's SQL editor (or via `supabase db push`)
-- after creating the project, before setting EXPO_PUBLIC_SUPABASE_URL/ANON_KEY.

create extension if not exists "pgcrypto";

-- One row per auth.users user, created automatically on signup.
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  username text unique,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "profiles are publicly readable"
  on public.profiles for select
  using (true);

create policy "users can update their own profile"
  on public.profiles for update
  using (auth.uid() = id);

-- Auto-create a profile row whenever a new auth user signs up.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, username)
  values (new.id, split_part(new.email, '@', 1));
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- Routes a user generated and chose to save.
create table if not exists public.routes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  grade smallint not null check (grade between 1 and 17),
  angle smallint not null check (angle between 0 and 70),
  image_data_uri text not null,
  created_at timestamptz not null default now()
);

create index if not exists routes_user_id_idx on public.routes (user_id);

alter table public.routes enable row level security;

create policy "routes are publicly readable"
  on public.routes for select
  using (true);

create policy "users can insert their own routes"
  on public.routes for insert
  with check (auth.uid() = user_id);

create policy "users can delete their own routes"
  on public.routes for delete
  using (auth.uid() = user_id);

-- One rating per (route, user).
create table if not exists public.ratings (
  id uuid primary key default gen_random_uuid(),
  route_id uuid not null references public.routes (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  stars smallint not null check (stars between 1 and 5),
  created_at timestamptz not null default now(),
  unique (route_id, user_id)
);

alter table public.ratings enable row level security;

create policy "ratings are publicly readable"
  on public.ratings for select
  using (true);

create policy "users can rate as themselves"
  on public.ratings for insert
  with check (auth.uid() = user_id);

create policy "users can update their own rating"
  on public.ratings for update
  using (auth.uid() = user_id);

create policy "users can delete their own rating"
  on public.ratings for delete
  using (auth.uid() = user_id);
