int main()
{
  int i;
  float result;
  float x = 10.0;
  float grad = 0.0;
  for (i = 0; i < 1000; i++)
  {
    grad = 2 * x;
    x = x - (0.1 * grad);
  }

  result = x * x;
  assert(result < 0.001);
  return 0;
}

