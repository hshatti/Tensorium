unit SortedMap;
{$ifdef FPC}
{$mode Delphi}
{$ModeSwitch advancedrecords}
{$endif}
{$pointerMath on}

interface

uses
  SysUtils, Generics.Defaults, nTypes;

type

  { TTools }

  TTools<T> = record
  type
    PT = ^T;
    TComparefunc = function(const a, b: T): SizeInt;
    class procedure QuickSort(Arr: PT; L, R: SizeInt; const Compare: TComparefunc; const Descending: boolean); static;
    class function BinSearch(const Arr: PT; const Val: T; R: SizeInt; Compare: TComparefunc): SizeInt; static;
  end;

  { TSortedMap }

  TSortedMap<K, V>= record
  type
    TKeyTools = TTools<K>;
  private
    // returns the position if key exists
    // otherwise returns a negative number := not(candidate position)
    class function defaultCompare(const a, b:K):SizeInt; static;
  public
    compare : TKeyTools.TComparefunc;
    Keys : TArray<k>;
    Values: TArray<V>;
    // returns the position of insert
    function lookup(const key:K):SizeInt;
    function updateOrInsert(const key:K; const value:V): SizeInt;
    function getValue(key:k):V;
    procedure setValue(key:k; value:V);
    function keyExists(const key:K):boolean;
    function count():SizeInt;
    // returns the position of delete or -1 if key was not found
    function deleteKey(const key:K):SizeInt;
    property Item[key:K]:V read getValue write setValue ;default;
  end;

implementation

class procedure TTools<T>.QuickSort(Arr: PT; L, R: SizeInt;
  const Compare: TComparefunc; const Descending: boolean);
var
  I, J, neg: SizeInt;
  P, Q: T;
begin
  if not Assigned(Arr) then exit;

  if descending then
    neg := -1
  else
    neg := 1;
  repeat
    I := L;
    J := R;
    P := Arr[(L + R) shr 1];
    repeat
      while (neg * Compare(P, Arr[I]) > 0) and (I <= R) do
        Inc(I);
      while (neg * Compare(P, Arr[J]) < 0) and (J >= 0) do
        Dec(J);
      if I <= J then
      begin
        Q := Arr[I];
        Arr[I] := Arr[J];
        Arr[J] := Q;
        I := I + 1;
        J := J - 1;
      end;
    until I > J;
    if J - L < R - I then
    begin
      if L < J then
        QuickSort(Arr, L, J, Compare, Descending);
      L := I;
    end
    else
    begin
      if I < R then
        QuickSort(Arr, I, R, Compare, Descending);
      R := J;
    end;
  until L >= R;
end;

class function TTools<T>.BinSearch(const Arr: PT; const Val: T; R: SizeInt;
  Compare: TComparefunc): SizeInt;
var
  L, I: SizeInt;
  CompareRes: IntPtr;isFound:boolean;
begin
  isFound := false;
  result:=-1;
  assert(assigned(compare), 'No <Compare> function assigned');
  // Use binary search.
  L := 0;
  R := R - 1;
  while (L<=R) do
  begin
    I := L + (R - L) shr 1;
    CompareRes := Compare(Val, Arr[I]);
    if (CompareRes>0) then
      L := I+1
    else begin
      R := I-1;
      if (CompareRes=0) then begin
         isFound := true;
//         if (Duplicates<>dupAccept) then
            L := I; // forces end of while loop
      end;
    end;
  end;
  if isFound then
    result := L
  else
    result := not(L);
end;

{ TSortedMap }

function TSortedMap<K, V>.lookup(const key: K): SizeInt;
var
  i:SizeInt;
  cmp: TKeyTools.TComparefunc;
begin
  if assigned(Compare) then cmp := compare else cmp:=defaultCompare;
  result := TKeyTools.BinSearch(pointer(keys), key, length(keys), cmp);
end;

class function TSortedMap<K, V>.defaultCompare(const a, b: K): SizeInt;
begin
  result := TComparer<K>.Default.Compare(a, b);
end;

function TSortedMap<K, V>.updateOrInsert(const key: K; const value: V): SizeInt;
begin
  result := lookup(key);
  if result < 0 then begin
    // we don't need to maintain a capacities,
    // pascal arrays does that under the hood
    // the only drawback is copying while shifting position make space
    insert(key, Keys, not result);
    insert(value, Values, not result);
  end else
    Values[result] := value

end;

function TSortedMap<K, V>.getValue(key: k): V;
var candidatePos: SizeInt;
begin
  result := default(V);
  candidatePos := lookup(key);
  assert(candidatePos>=0, 'Key does not exits!');
  if candidatePos>=0 then
    result := Values[candidatePos]
end;

procedure TSortedMap<K, V>.setValue(key: k; value: V);
var candidatePos :SizeInt;
begin
  candidatePos := lookup(key);
  if candidatePos < 0 then begin
    // we don't need to maintain a capacities,
    // pascal arrays does that under the hood
    // the only drawback is copying while shifting position make space
    insert(key, Keys, not candidatePos);
    insert(value, Values, not candidatePos);
  end else
    Values[candidatePos] := value
end;

function TSortedMap<K, V>.keyExists(const key: K): boolean;
begin
  result := lookup(key)>=0
end;

function TSortedMap<K, V>.count(): SizeInt;
begin
  result:= length(Keys)
end;

function TSortedMap<K, V>.deleteKey(const key: K): SizeInt;
begin
  result := lookup(key);
  if result>=0 then begin
    delete(Keys, result, 1);
    delete(Values, result, 1);
  end;
end;


end.

