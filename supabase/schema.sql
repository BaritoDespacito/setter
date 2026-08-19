-- Setter frontend schema: profiles, saved routes, ratings.
-- Safe to re-run in full any time (every statement is idempotent) - run it in the
-- Supabase project's SQL editor (or via `supabase db push`) after creating the
-- project, before setting EXPO_PUBLIC_SUPABASE_URL/ANON_KEY, and again whenever
-- this file changes.

create extension if not exists "pgcrypto";

-- One row per auth.users user, created automatically on signup.
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  username text unique,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "profiles are publicly readable" on public.profiles;
create policy "profiles are publicly readable"
  on public.profiles for select
  using (true);

drop policy if exists "users can update their own profile" on public.profiles;
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

-- Routes a user generated and chose to save. Images live in Supabase Storage (see
-- the route-images bucket below), not inline here - image_url just points at them.
-- Storing the ~1.2MB base64 PNG directly in this column (the original design) made
-- the routes list query pull megabytes of text per page load; a URL keeps rows tiny
-- and lets the client lazy-load/cache images normally.
create table if not exists public.routes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  grade smallint not null check (grade between 1 and 17),
  angle smallint not null check (angle between 0 and 70),
  image_url text not null,
  created_at timestamptz not null default now()
);

-- Migration for projects that already ran an older version of this schema (column
-- used to be called image_data_uri and held inline base64 data URIs).
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'routes' and column_name = 'image_data_uri'
  ) then
    alter table public.routes rename column image_data_uri to image_url;
  end if;
end $$;

create index if not exists routes_user_id_idx on public.routes (user_id);

alter table public.routes enable row level security;

drop policy if exists "routes are publicly readable" on public.routes;
create policy "routes are publicly readable"
  on public.routes for select
  using (true);

drop policy if exists "users can insert their own routes" on public.routes;
create policy "users can insert their own routes"
  on public.routes for insert
  with check (auth.uid() = user_id);

drop policy if exists "users can delete their own routes" on public.routes;
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

drop policy if exists "ratings are publicly readable" on public.ratings;
create policy "ratings are publicly readable"
  on public.ratings for select
  using (true);

drop policy if exists "users can rate as themselves" on public.ratings;
create policy "users can rate as themselves"
  on public.ratings for insert
  with check (auth.uid() = user_id);

drop policy if exists "users can update their own rating" on public.ratings;
create policy "users can update their own rating"
  on public.ratings for update
  using (auth.uid() = user_id);

drop policy if exists "users can delete their own rating" on public.ratings;
create policy "users can delete their own rating"
  on public.ratings for delete
  using (auth.uid() = user_id);

-- Route image storage. Objects are stored as "<user_id>/<route_id>.png" - the
-- policies below key off that first path segment to scope writes/deletes to the
-- uploading user, while reads are public (route images are meant to be shareable,
-- same as the routes table itself).
insert into storage.buckets (id, name, public)
values ('route-images', 'route-images', true)
on conflict (id) do nothing;

drop policy if exists "route images are publicly readable" on storage.objects;
create policy "route images are publicly readable"
  on storage.objects for select
  using (bucket_id = 'route-images');

drop policy if exists "users can upload their own route images" on storage.objects;
create policy "users can upload their own route images"
  on storage.objects for insert
  with check (bucket_id = 'route-images' and auth.uid()::text = (storage.foldername(name))[1]);

drop policy if exists "users can delete their own route images" on storage.objects;
create policy "users can delete their own route images"
  on storage.objects for delete
  using (bucket_id = 'route-images' and auth.uid()::text = (storage.foldername(name))[1]);
